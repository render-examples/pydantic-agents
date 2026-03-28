"""FastAPI application for Ask Render Anything Assistant."""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logfire
from opentelemetry import trace

import asyncio

from backend.config import settings
from backend.models import (
    QuestionRequest,
    AnswerResponse,
    HealthCheck,
    ProgressUpdate,
    PipelineStageResult,
)
from backend.database import vector_store
from backend.observability import pipeline_trace, track_pipeline_metrics, calculate_anthropic_cost
from backend.prices import load_prices
from backend.pipeline import (
    embed_question,
    retrieve_documents,
    generate_answer,
    stream_answer,
    extract_claims,
    verify_claims,
    check_accuracy,
    evaluate_quality,
    quality_gate_decision,
)
from backend.api.logs import fetch_logfire_logs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    
    # Startup
    logfire.info("Starting Ask Render Anything Assistant")
    await vector_store.initialize()
    await load_prices()
    logfire.info("Application started successfully")
    
    yield
    
    # Shutdown
    logfire.info("Shutting down Ask Render Anything Assistant")
    await vector_store.close()
    logfire.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Ask Render Anything Assistant",
    description="Production-grade AI pipeline with observable AI using Pydantic AI, Logfire, and Render",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with Logfire for automatic HTTP tracing
logfire.instrument_fastapi(app)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "name": "Ask Render Anything Assistant",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    
    db_healthy = await vector_store.health_check()
    
    return HealthCheck(
        status="healthy" if db_healthy else "degraded",
        database_connected=db_healthy,
        logfire_enabled=True
    )


async def execute_pipeline(question: str, session_id: str = None) -> AnswerResponse:
    """Execute the full 8-stage pipeline."""
    
    async with pipeline_trace(question) as context:
        stages = []
        total_cost = 0.0
        pipeline_start = time.time()
        
        # Stage 1: Question Embedding
        embed_result = await embed_question(question)
        stage_result = PipelineStageResult(
            stage="question_embedding",
            success=True,
            duration_ms=0,  # Tracked by decorator
            cost_usd=embed_result["cost_usd"],
            tokens_used=embed_result["tokens"],
            model=settings.embedding_model,
            metadata={"embedding_dimensions": len(embed_result["embedding"])}
        )
        stages.append(stage_result)
        total_cost += embed_result["cost_usd"]
        
        # Stage 2: RAG Retrieval (with multi-query expansion for broad questions)
        retrieval_result = await retrieve_documents(embed_result["embedding"], original_question=question)
        stage_result = PipelineStageResult(
            stage="rag_retrieval",
            success=True,
            duration_ms=0,
            cost_usd=retrieval_result["cost_usd"],
            model=settings.query_expansion_model,
            metadata={
                "documents_retrieved": len(retrieval_result["documents"]),
                "queries_expanded": retrieval_result.get("queries_count", 1)
            }
        )
        stages.append(stage_result)
        total_cost += retrieval_result["cost_usd"]
        
        documents = retrieval_result["documents"]
        
        # Iterative quality refinement loop
        current_iteration = 1
        feedback = None
        answer_text = ""
        verified_claims = []
        accuracy_score = 0
        evaluations = []
        average_score = 0
        
        while current_iteration <= settings.max_iterations:
            logfire.info(f"Starting iteration {current_iteration}")
            
            # Stage 3: Answer Generation
            gen_result = await generate_answer(question, documents, feedback)
            stage_result = PipelineStageResult(
                stage=f"answer_generation_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=gen_result["cost_usd"],
                tokens_used=gen_result["input_tokens"] + gen_result["output_tokens"],
                model=settings.answer_model,
                metadata={
                    "answer_length": len(gen_result["answer"]),
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += gen_result["cost_usd"]
            answer_text = gen_result["answer"]

            # Stage 4: Claims Extraction
            claims_result = await extract_claims(answer_text)
            claims_count = len(claims_result["claims"])
            stage_result = PipelineStageResult(
                stage=f"claims_extraction_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=claims_result["cost_usd"],
                tokens_used=claims_result["input_tokens"] + claims_result["output_tokens"],
                model=settings.claims_model,
                metadata={
                    "claims_extracted": claims_count,
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += claims_result["cost_usd"]
            
            # Stage 5: Claims Verification
            verification_result = await verify_claims(claims_result["claims"])
            verified_claims = verification_result["verified_claims"]
            verification_rate = verification_result["verification_rate"] * 100
            verified_count = len([c for c in verified_claims if c.verified])
            stage_result = PipelineStageResult(
                stage=f"claims_verification_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=verification_result["cost_usd"],
                model=settings.embedding_model,
                metadata={
                    "claims_verified": verified_count,
                    "total_claims": len(verified_claims),
                    "verification_rate": f"{verification_rate:.0f}%",
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            total_cost += verification_result["cost_usd"]

            # Stages 6 + 7: Technical Accuracy and Quality Evaluation (run in parallel)
            accuracy_result, eval_result = await asyncio.gather(
                check_accuracy(answer_text, verified_claims),
                evaluate_quality(question, answer_text, documents),
            )
            accuracy_score = accuracy_result["accuracy_score"]
            evaluations = eval_result["evaluations"]
            average_score = eval_result["average_score"]
            stages.append(PipelineStageResult(
                stage=f"technical_accuracy_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=accuracy_result["cost_usd"],
                tokens_used=accuracy_result["input_tokens"] + accuracy_result["output_tokens"],
                model=settings.accuracy_model,
                metadata={
                    "accuracy_score": accuracy_score,
                    "iteration": current_iteration
                }
            ))
            total_cost += accuracy_result["cost_usd"]
            stages.append(PipelineStageResult(
                stage=f"quality_evaluation_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=eval_result["cost_usd"],
                model=f"{settings.eval_model_openai} + {settings.eval_model_anthropic}",
                metadata={
                    "quality_score": f"{average_score:.1f}",
                    "openai_score": evaluations[0].score if len(evaluations) > 0 else None,
                    "claude_score": evaluations[1].score if len(evaluations) > 1 else None,
                    "agreement": eval_result.get("agreement_level", "unknown"),
                    "iteration": current_iteration
                }
            ))
            total_cost += eval_result["cost_usd"]
            
            # Stage 8: Quality Gate
            gate_result = await quality_gate_decision(
                average_score=average_score,
                evaluations=evaluations,
                accuracy_score=accuracy_score,
                current_iteration=current_iteration,
                errors=accuracy_result["errors"],
                corrections=accuracy_result["corrections"]
            )
            stage_result = PipelineStageResult(
                stage=f"quality_gate_iter_{current_iteration}",
                success=True,
                duration_ms=0,
                cost_usd=0.0,
                metadata={
                    "should_iterate": gate_result["should_iterate"],
                    "reason": gate_result["reason"],
                    "iteration": current_iteration
                }
            )
            stages.append(stage_result)
            
            # Check if we should iterate
            if not gate_result["should_iterate"]:
                logfire.info(f"Quality gate passed: {gate_result['reason']}")
                break
            
            logfire.info(f"Quality gate requires iteration: {gate_result['reason']}")
            feedback = gate_result["feedback"]
            current_iteration += 1
        
        # Calculate total duration
        total_duration_ms = (time.time() - pipeline_start) * 1000
        
        # Create response
        response = AnswerResponse(
            question=question,
            answer=answer_text,
            sources=documents,
            claims=verified_claims,
            quality_score=average_score,
            accuracy_score=accuracy_score,
            evaluations=evaluations,
            iterations=current_iteration,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            stages=stages,
            session_id=session_id
        )
        
        # Track custom metrics for observability
        track_pipeline_metrics(
            question=question,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            quality_score=average_score,
            accuracy_score=accuracy_score,
            iterations=current_iteration,
            session_id=session_id
        )
        
        return response


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get a high-quality answer.
    
    This endpoint executes the full 8-stage pipeline including:
    - Question embedding
    - RAG document retrieval
    - Answer generation
    - Claims extraction & verification
    - Technical accuracy check
    - Dual-model quality evaluation
    - Quality gate decision (with iterative refinement)
    """
    
    # Create session-level span for end-to-end user journey tracking
    with logfire.span(
        "user_session.qa_request",
        session_id=request.session_id or "anonymous",
        question=request.question[:100],  # First 100 chars for preview
        question_length=len(request.question)
    ):
        try:
            logfire.info(
                "Received question",
                session_id=request.session_id or "anonymous",
                question_length=len(request.question)
            )
            
            response = await execute_pipeline(request.question, request.session_id)
            
            logfire.info(
                "Question answered successfully",
                session_id=request.session_id or "anonymous",
                duration_ms=response.total_duration_ms,
                cost_usd=response.total_cost,
                quality_score=response.quality_score,
                iterations=response.iterations
            )
            
            return response
        
        except Exception as e:
            logfire.error(
                "Error processing question",
                session_id=request.session_id or "anonymous",
                error=str(e),
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


def _progress_event(stage: str, status: str, message: str, progress: float, cost_so_far: float) -> str:
    return f"data: {json.dumps(ProgressUpdate(stage=stage, status=status, message=message, progress=progress, cost_so_far=cost_so_far).model_dump())}\n\n"


async def _persist_session(
    question: str,
    answer_text: str,
    documents: list,
    verified_claims: list,
    evaluations: list,
    average_score: float,
    iterations: int,
    total_cost: float,
    total_duration_ms: float,
    stages: list,
) -> str | None:
    try:
        current_span = trace.get_current_span()
        trace_id = None
        if current_span and current_span.get_span_context().is_valid:
            trace_id = format(current_span.get_span_context().trace_id, '032x')
        saved_id = await vector_store.save_session(
            question=question,
            answer=answer_text,
            sources=[doc.model_dump() for doc in documents],
            claims=[claim.model_dump() for claim in verified_claims],
            evaluations=[ev.model_dump() for ev in evaluations],
            quality_score=average_score,
            iterations=iterations,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            trace_id=trace_id,
            stages=[s.model_dump() for s in stages]
        )
        logfire.info(f"Saved session to database: {saved_id}", trace_id=trace_id)
        return saved_id
    except Exception as e:
        logfire.error(f"Failed to save session: {e}")
        return None


async def pipeline_generator(question: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """Generator for Server-Sent Events during pipeline execution."""
    try:
        async with pipeline_trace(question):
            pipeline_start = time.time()
            total_cost = 0.0
            stages = []

            # Stage 1: Embedding
            yield _progress_event("question_embedding", "started", "Embedding your question...", 5, total_cost)
            stage_start = time.time()
            embed_result = await embed_question(question)
            total_cost += embed_result["cost_usd"]
            stages.append(PipelineStageResult(
                stage="question_embedding",
                success=True,
                duration_ms=(time.time() - stage_start) * 1000,
                cost_usd=embed_result["cost_usd"],
                model=settings.embedding_model,
                metadata={"embedding_dimensions": len(embed_result["embedding"])}
            ))
            yield _progress_event("question_embedding", "completed", "Question embedded", 12.5, total_cost)

            # Stage 2: Retrieval
            yield _progress_event("rag_retrieval", "started", "Searching documentation...", 15, total_cost)
            stage_start = time.time()
            retrieval_result = await retrieve_documents(embed_result["embedding"], original_question=question)
            total_cost += retrieval_result["cost_usd"]
            documents = retrieval_result["documents"]
            stages.append(PipelineStageResult(
                stage="rag_retrieval",
                success=True,
                duration_ms=(time.time() - stage_start) * 1000,
                cost_usd=retrieval_result["cost_usd"],
                model=settings.query_expansion_model,
                metadata={
                    "documents_retrieved": len(documents),
                    "queries_expanded": retrieval_result.get("queries_count", 1)
                }
            ))
            yield _progress_event("rag_retrieval", "completed", f"Found {len(documents)} relevant documents", 25, total_cost)

            # Stages 3–8: iterative refinement
            # Each iteration gets an equal share of the 25%–85% progress band.
            iter_span = 60 / settings.max_iterations
            feedback = None
            answer_text = ""
            verified_claims = []
            accuracy_score = 0
            evaluations = []
            average_score = 0.0
            final_iteration = settings.max_iterations

            for iteration in range(1, settings.max_iterations + 1):
                p = 25 + (iteration - 1) * iter_span  # progress baseline for this iteration

                # Stage 3: Answer generation (streamed token-by-token)
                yield _progress_event(f"generation_iter_{iteration}", "started", f"Generating answer (iteration {iteration})...", min(p + 0.05 * iter_span, 95), total_cost)
                stage_start = time.time()
                answer_text = ""
                gen_usage = None
                async for delta, usage in stream_answer(question, documents, feedback):
                    if delta:
                        answer_text += delta
                        yield f"data: {json.dumps({'type': 'answer_token', 'delta': delta})}\n\n"
                    elif usage is not None:
                        gen_usage = usage
                input_tokens = (gen_usage.request_tokens or 0) if gen_usage else 0
                output_tokens = (gen_usage.response_tokens or 0) if gen_usage else 0
                gen_cost = calculate_anthropic_cost(input_tokens, output_tokens, settings.answer_model)
                total_cost += gen_cost
                stages.append(PipelineStageResult(
                    stage=f"answer_generation_iter_{iteration}",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    cost_usd=gen_cost,
                    tokens_used=input_tokens + output_tokens,
                    model=settings.answer_model,
                    metadata={"answer_length": len(answer_text), "iteration": iteration}
                ))
                yield _progress_event(f"generation_iter_{iteration}", "completed", "Answer generated", min(p + 0.20 * iter_span, 95), total_cost)

                # Stage 4: Claims extraction
                yield _progress_event(f"claims_iter_{iteration}", "started", "Extracting factual claims...", min(p + 0.30 * iter_span, 95), total_cost)
                stage_start = time.time()
                claims_result = await extract_claims(answer_text)
                total_cost += claims_result["cost_usd"]
                claims_count = len(claims_result["claims"])
                stages.append(PipelineStageResult(
                    stage=f"claims_extraction_iter_{iteration}",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    cost_usd=claims_result["cost_usd"],
                    model=settings.claims_model,
                    metadata={"claims_extracted": claims_count, "iteration": iteration}
                ))
                yield _progress_event(f"claims_iter_{iteration}", "completed", f"Extracted {claims_count} claims", min(p + 0.40 * iter_span, 95), total_cost)

                # Stage 5: Claims verification
                yield _progress_event(f"verification_iter_{iteration}", "started", "Verifying claims...", min(p + 0.50 * iter_span, 95), total_cost)
                stage_start = time.time()
                verification_result = await verify_claims(claims_result["claims"])
                total_cost += verification_result["cost_usd"]
                verified_claims = verification_result["verified_claims"]
                verification_rate = verification_result["verification_rate"] * 100
                verified_count = len([c for c in verified_claims if c.verified])
                stages.append(PipelineStageResult(
                    stage=f"claims_verification_iter_{iteration}",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    cost_usd=verification_result["cost_usd"],
                    model=settings.embedding_model,
                    metadata={
                        "claims_verified": verified_count,
                        "total_claims": len(verified_claims),
                        "verification_rate": f"{verification_rate:.0f}%",
                        "iteration": iteration
                    }
                ))
                yield _progress_event(f"verification_iter_{iteration}", "completed", f"{verification_rate:.0f}% claims verified", min(p + 0.60 * iter_span, 95), total_cost)

                # Stages 6 + 7: Accuracy and quality evaluation (parallel)
                yield _progress_event(f"accuracy_iter_{iteration}", "started", "Checking accuracy & quality in parallel...", min(p + 0.70 * iter_span, 95), total_cost)
                stage_start = time.time()
                accuracy_result, eval_result = await asyncio.gather(
                    check_accuracy(answer_text, verified_claims),
                    evaluate_quality(question, answer_text, documents),
                )
                parallel_duration = (time.time() - stage_start) * 1000

                accuracy_score = accuracy_result["accuracy_score"]
                total_cost += accuracy_result["cost_usd"]
                stages.append(PipelineStageResult(
                    stage=f"accuracy_check_iter_{iteration}",
                    success=True,
                    duration_ms=parallel_duration,
                    cost_usd=accuracy_result["cost_usd"],
                    model=settings.accuracy_model,
                    metadata={"accuracy_score": accuracy_score, "iteration": iteration}
                ))
                yield _progress_event(f"accuracy_iter_{iteration}", "completed", f"Accuracy score: {accuracy_score}/100", min(p + 0.80 * iter_span, 95), total_cost)

                evaluations = eval_result["evaluations"]
                average_score = eval_result["average_score"]
                total_cost += eval_result["cost_usd"]
                stages.append(PipelineStageResult(
                    stage=f"quality_evaluation_iter_{iteration}",
                    success=True,
                    duration_ms=parallel_duration,
                    cost_usd=eval_result["cost_usd"],
                    model=f"{settings.eval_model_openai} + {settings.eval_model_anthropic}",
                    metadata={
                        "quality_score": f"{average_score:.1f}",
                        "openai_score": evaluations[0].score if evaluations else None,
                        "claude_score": evaluations[1].score if len(evaluations) > 1 else None,
                        "agreement": eval_result.get("agreement_level", "unknown"),
                        "iteration": iteration
                    }
                ))
                yield _progress_event(f"evaluation_iter_{iteration}", "completed", f"Quality score: {average_score:.1f}/100", min(p + 0.90 * iter_span, 95), total_cost)

                # Stage 8: Quality gate
                yield _progress_event(f"quality_gate_iter_{iteration}", "started", "Checking quality gate...", min(p + 0.95 * iter_span, 95), total_cost)
                gate_result = await quality_gate_decision(
                    average_score=average_score,
                    evaluations=evaluations,
                    accuracy_score=accuracy_score,
                    current_iteration=iteration,
                    errors=accuracy_result["errors"],
                    corrections=accuracy_result["corrections"]
                )

                if gate_result["should_iterate"]:
                    yield _progress_event(f"quality_gate_iter_{iteration}", "completed", f"Refining answer... ({gate_result['reason']})", min(p + iter_span, 85), total_cost)
                    feedback = gate_result["feedback"]
                    continue

                yield _progress_event(f"quality_gate_iter_{iteration}", "completed", "Quality gate passed!", 95, total_cost)
                final_iteration = iteration
                break

            # Send final result — reached here by gate passing (break) or exhausting iterations.
            total_duration_ms = (time.time() - pipeline_start) * 1000
            response = AnswerResponse(
                question=question,
                answer=answer_text,
                sources=documents,
                claims=verified_claims,
                quality_score=average_score,
                accuracy_score=accuracy_score,
                evaluations=evaluations,
                iterations=final_iteration,
                total_cost=total_cost,
                total_duration_ms=total_duration_ms,
                stages=stages,
                session_id=session_id
            )
            response.session_id = await _persist_session(
                question=question,
                answer_text=answer_text,
                documents=documents,
                verified_claims=verified_claims,
                evaluations=evaluations,
                average_score=average_score,
                iterations=final_iteration,
                total_cost=total_cost,
                total_duration_ms=total_duration_ms,
                stages=stages,
            )
            yield f"data: {json.dumps({'type': 'complete', 'result': response.model_dump(mode='json')})}\n\n"

    except Exception as e:
        logfire.error(f"Error in pipeline generator: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/ask/stream", tags=["Q&A"])
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question with real-time progress updates via Server-Sent Events.
    
    This endpoint provides the same functionality as /ask but streams
    progress updates as the pipeline executes.
    """
    
    logfire.info(f"Received streaming question: {request.question}")
    
    return StreamingResponse(
        pipeline_generator(request.question, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/stats", tags=["Admin"])
async def get_stats():
    """Get database statistics."""
    
    doc_count = await vector_store.get_document_count()
    
    return {
        "document_count": doc_count,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "rag_top_k": settings.rag_top_k,
        "quality_threshold": settings.quality_threshold,
        "max_iterations": settings.max_iterations
    }


@app.get("/history", tags=["Q&A"])
async def get_history(limit: int = 20):
    """
    Get recent Q&A sessions.
    
    Args:
        limit: Maximum number of sessions to return (default: 20, max: 100)
    """
    
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    
    sessions = await vector_store.get_recent_sessions(limit=limit)
    
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@app.get("/history/{session_id}", tags=["Q&A"])
async def get_session(session_id: str):
    """
    Get a specific Q&A session by ID.
    
    Args:
        session_id: The UUID of the session
    """
    
    session = await vector_store.get_session_by_id(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@app.delete("/history/{session_id}", tags=["Q&A"])
async def delete_session(session_id: str):
    """
    Delete a specific Q&A session by ID.
    
    Args:
        session_id: The UUID of the session to delete
    """
    
    deleted = await vector_store.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "success": True,
        "message": "Session deleted successfully",
        "session_id": session_id
    }


@app.delete("/history", tags=["Q&A"])
async def clear_all_history():
    """
    Delete all Q&A sessions from history.
    
    This action cannot be undone.
    """
    
    deleted_count = await vector_store.delete_all_sessions()
    
    return {
        "success": True,
        "message": f"Deleted {deleted_count} sessions",
        "count": deleted_count
    }


@app.get("/sessions/{session_id}/logs", tags=["Observability"])
async def get_session_logs(session_id: str):
    """
    Fetch Logfire logs for a specific Q&A session.
    
    Returns detailed observability logs from Logfire for the given session,
    including all spans, traces, and metrics captured during execution.
    """
    # Get session from database to retrieve trace_id
    session = await vector_store.get_session_by_id(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    trace_id = session.get('trace_id')
    if not trace_id:
        raise HTTPException(
            status_code=404, 
            detail="Trace ID not available for this session (may be from before trace logging was enabled)"
        )
    
    # Fetch logs from Logfire API
    try:
        logs_data = await fetch_logfire_logs(trace_id)
        return logs_data
    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Unexpected error fetching logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )

