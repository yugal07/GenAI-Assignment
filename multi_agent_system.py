from __future__ import annotations

import os
import sys
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


class ResumeState(TypedDict, total=False):
    resume_text: str
    target_role: str
    parsed_profile: str
    analysis: str
    job_match: str
    recommendations: str


_DEFAULT_GOOGLE_API_KEY = "AIzaSyAwiOD6xXL4p1bDb4k9xDeIHB10PdjcG3M"


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY") or _DEFAULT_GOOGLE_API_KEY
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature,
        google_api_key=api_key,
    )


def parser_agent(state: ResumeState) -> ResumeState:
    print("\n[ParserAgent] Extracting structured profile from resume...")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert resume parser. Extract structured information from the "
         "given resume. Return clean sections for:\n"
         " - Candidate Name & Contact\n"
         " - Summary\n"
         " - Skills (grouped by category if possible)\n"
         " - Work Experience (company, role, dates, key achievements)\n"
         " - Education\n"
         " - Certifications / Projects\n"
         "Be concise and faithful to the source. Do not invent information."),
        ("human", "Resume:\n{resume_text}")
    ])
    chain = prompt | get_llm(0.0) | StrOutputParser()
    parsed = chain.invoke({"resume_text": state["resume_text"]})
    return {"parsed_profile": parsed}


def analyzer_agent(state: ResumeState) -> ResumeState:
    print("[AnalyzerAgent] Analyzing strengths and weaknesses...")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior career coach. Given the structured profile below, "
         "produce an honest assessment covering:\n"
         " 1. Top 3 Strengths\n"
         " 2. Top 3 Weaknesses / Gaps\n"
         " 3. Writing & Formatting quality (clarity, impact verbs, quantified results)\n"
         " 4. Overall resume quality score (1-10) with a one-line justification."),
        ("human", "Structured Profile:\n{parsed_profile}")
    ])
    chain = prompt | get_llm(0.3) | StrOutputParser()
    analysis = chain.invoke({"parsed_profile": state["parsed_profile"]})
    return {"analysis": analysis}


def job_matcher_agent(state: ResumeState) -> ResumeState:
    print("[JobMatcherAgent] Matching profile against target role...")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a technical recruiter. Compare the candidate's profile to the target "
         "role. Produce:\n"
         " - Matching skills / experience (bullet list)\n"
         " - Missing or under-represented skills\n"
         " - Transferable skills worth highlighting\n"
         " - A match percentage (0-100%) with a one-sentence justification."),
        ("human",
         "Target Role:\n{target_role}\n\n"
         "Candidate Profile:\n{parsed_profile}\n\n"
         "Prior Analysis:\n{analysis}")
    ])
    chain = prompt | get_llm(0.2) | StrOutputParser()
    match = chain.invoke({
        "target_role": state["target_role"],
        "parsed_profile": state["parsed_profile"],
        "analysis": state["analysis"],
    })
    return {"job_match": match}


def recommender_agent(state: ResumeState) -> ResumeState:
    print("[RecommenderAgent] Generating final recommendations...\n")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a resume improvement specialist. Using the prior analysis and "
         "job-fit report, produce a final actionable plan:\n"
         " 1. Executive Summary (3-4 sentences)\n"
         " 2. Top 5 Prioritized Improvements (each with a concrete rewrite example "
         "    where relevant)\n"
         " 3. Suggested skills/certifications to acquire for the target role\n"
         " 4. Final Fit Verdict: Strong Fit / Moderate Fit / Weak Fit + reasoning."),
        ("human",
         "Target Role:\n{target_role}\n\n"
         "Analysis:\n{analysis}\n\n"
         "Job Match Report:\n{job_match}")
    ])
    chain = prompt | get_llm(0.4) | StrOutputParser()
    recs = chain.invoke({
        "target_role": state["target_role"],
        "analysis": state["analysis"],
        "job_match": state["job_match"],
    })
    return {"recommendations": recs}


def build_graph():
    graph = StateGraph(ResumeState)

    graph.add_node("parser", parser_agent)
    graph.add_node("analyzer", analyzer_agent)
    graph.add_node("job_matcher", job_matcher_agent)
    graph.add_node("recommender", recommender_agent)

    graph.add_edge(START, "parser")
    graph.add_edge("parser", "analyzer")
    graph.add_edge("analyzer", "job_matcher")
    graph.add_edge("job_matcher", "recommender")
    graph.add_edge("recommender", END)

    return graph.compile()


def read_multiline(prompt: str) -> str:
    print(prompt)
    print("(paste content; type 'END' on its own line when finished)\n")
    lines: list[str] = []
    for line in sys.stdin:
        if line.strip() == "END":
            break
        lines.append(line)
    return "".join(lines).strip()


def get_user_input() -> tuple[str, str]:
    resume_text = read_multiline(">>> Paste the resume text:")
    if not resume_text:
        raise ValueError("Resume text cannot be empty.")

    print("\n>>> Enter the target job role or paste a short job description:")
    target_role = input("> ").strip()
    if not target_role:
        raise ValueError("Target role cannot be empty.")

    return resume_text, target_role


def main() -> None:
    print("=" * 70)
    print(" Multi-Agent Resume Reviewer (LangChain + LangGraph) ")
    print("=" * 70)

    try:
        resume_text, target_role = get_user_input()
    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)

    app = build_graph()

    initial_state: ResumeState = {
        "resume_text": resume_text,
        "target_role": target_role,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 70)
    print(" PARSED PROFILE ".center(70, "="))
    print("=" * 70)
    print(final_state.get("parsed_profile", "").strip())

    print("\n" + "=" * 70)
    print(" ANALYSIS ".center(70, "="))
    print("=" * 70)
    print(final_state.get("analysis", "").strip())

    print("\n" + "=" * 70)
    print(" JOB-FIT REPORT ".center(70, "="))
    print("=" * 70)
    print(final_state.get("job_match", "").strip())

    print("\n" + "=" * 70)
    print(" FINAL RECOMMENDATIONS ".center(70, "="))
    print("=" * 70)
    print(final_state.get("recommendations", "").strip())
    print()


if __name__ == "__main__":
    main()
