//! Prompt templates and the refined-answer extractor.
//!
//! These strings are ported **verbatim** from the original Python implementation
//! (`framework/fusion_core.py`) so the debate behaves identically. Do not reword
//! them casually — the unit tests pin the exact output, and changes alter model
//! behavior. `IndexMap` is used for peer/synthesis inputs so ordering is
//! deterministic (matching Python's insertion-ordered dicts).

use indexmap::IndexMap;
use once_cell::sync::Lazy;
use regex::Regex;

/// Role-aware system prompt for a debating agent.
///
/// Mirrors `Fusion._build_system_prompt`: `role` defaults to the agent name when
/// unset.
pub fn build_system_prompt(name: &str, role: Option<&str>) -> String {
    let role = role.unwrap_or(name);
    format!(
        "You are {name}, specialized in {role}. \
Follow instructions carefully, avoid fabrications, and provide step-by-step, verifiable reasoning when asked."
    )
}

/// System prompt for the synthesizer agent (verbatim from the Python).
pub fn synthesizer_system_prompt() -> &'static str {
    "You are the synthesizer. Merge inputs into the single best answer, \
maximizing clarity, correctness, and completeness."
}

/// Prompt template for the initial generation step.
pub fn build_initial_prompt(query: &str, paper_mode: bool) -> String {
    if paper_mode {
        format!(
            "Paper Writing Mode. Task: Compose a clear, well-structured scholarly response. \
Structure your answer with: Abstract, Introduction, Methods/Approach, Results/Findings, Discussion, Conclusion, and References (if applicable).\n\n\
Prompt: {query}"
        )
    } else {
        format!("Task: Provide the best possible answer to the user's query.\n\nQuery: {query}")
    }
}

/// Prompt template for the critical-review + refinement step.
///
/// `other_responses` is rendered as `[Name]\nresponse` blocks joined by blank
/// lines, in iteration order.
pub fn build_review_prompt(
    query: &str,
    self_response: &str,
    other_responses: &IndexMap<String, String>,
    paper_mode: bool,
) -> String {
    let others_str = other_responses
        .iter()
        .map(|(name, resp)| format!("[{name}]\n{resp}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    let mode_line = if paper_mode {
        "Maintain the requested academic structure. "
    } else {
        ""
    };
    format!(
        "You will review responses from other agents and refine your own. {mode_line}\
Instructions:\n\
1) Identify factual errors, logical gaps, and unclear explanations in others' responses.\n\
2) Suggest concrete improvements and corrections.\n\
3) Produce your refined answer that integrates the best ideas and fixes flaws.\n\n\
Original Query:\n{query}\n\n\
Your Previous Answer:\n{self_response}\n\n\
Other Agents' Answers:\n{others_str}\n\n\
Output format:\n\
- Critique: <your short critique>\n\
- Refined Answer: <your improved answer>\n"
    )
}

/// Prompt template for the final synthesizer call.
pub fn build_synthesis_prompt(
    query: &str,
    agent_outputs: &IndexMap<String, String>,
    paper_mode: bool,
) -> String {
    let outputs_joined = agent_outputs
        .iter()
        .map(|(name, content)| format!("[{name}]\n{content}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    let mode_line = if paper_mode {
        "Ensure academic structure (Abstract, Introduction, Methods, Results, Discussion, Conclusion). "
    } else {
        ""
    };
    format!(
        "You are the synthesizer. {mode_line}\
Merge the following agent answers into a single best response.\
 Be precise, cite assumptions, and avoid contradictions. If there is disagreement, resolve it with reasoning or present consensus with justification.\n\n\
Original Query:\n{query}\n\n\
Agent Answers:\n{outputs_joined}\n\n\
Final Answer:"
    )
}

/// Match a "Refined Answer:" header only at the start of a line, tolerating
/// leading bullets/markdown bold, so an in-sentence mention inside a critique
/// ("...my refined answer:...") doesn't get mistaken for the header.
static REFINED_MARKER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?im)^[ \t]*[-*]?[ \t]*\**[ \t]*refined answer[ \t]*\**[ \t]*:[ \t]*\**[ \t]*")
        .expect("refined-answer regex is valid")
});

/// Heuristic to extract the refined-answer block from an agent's output.
///
/// Uses the **last** header match so a real "Refined Answer:" block wins over an
/// earlier in-critique reference. Falls back to the trimmed whole text when no
/// header is present. (The Python `[MOCK]` passthrough is intentionally dropped
/// — there is no mock mode.)
pub fn extract_refined(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if let Some(m) = REFINED_MARKER_RE.find_iter(text).last() {
        text[m.end()..].trim().to_string()
    } else {
        text.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn extract_refined_plain_marker() {
        let text = "- Critique: looks good\n- Refined Answer: 42 is the answer";
        assert_eq!(extract_refined(text), "42 is the answer");
    }

    #[test]
    fn extract_refined_markdown_bold() {
        let text = "Critique: ok\n**Refined Answer:** bold answer";
        assert_eq!(extract_refined(text), "bold answer");
    }

    #[test]
    fn extract_refined_no_marker_returns_trimmed() {
        let text = "  just an answer with no header  ";
        assert_eq!(extract_refined(text), "just an answer with no header");
    }

    #[test]
    fn extract_refined_uses_last_marker() {
        // An in-critique mention must lose to the real trailing header.
        let text = "Critique: my refined answer: was weak.\nRefined Answer: the strong one";
        assert_eq!(extract_refined(text), "the strong one");
    }

    #[test]
    fn extract_refined_empty() {
        assert_eq!(extract_refined(""), "");
    }

    #[test]
    fn initial_prompt_normal_vs_paper() {
        assert!(build_initial_prompt("Q", false).starts_with("Task: Provide the best"));
        assert!(build_initial_prompt("Q", true).starts_with("Paper Writing Mode."));
        assert!(build_initial_prompt("Q", false).ends_with("Query: Q"));
    }

    #[test]
    fn review_prompt_includes_peers_and_mode_line() {
        let others = map(&[("Alice", "a-resp"), ("Bob", "b-resp")]);
        let p = build_review_prompt("Q", "mine", &others, false);
        assert!(p.contains("[Alice]\na-resp"));
        assert!(p.contains("[Bob]\nb-resp"));
        assert!(p.contains("Your Previous Answer:\nmine"));
        assert!(!p.contains("Maintain the requested academic structure."));
        let pp = build_review_prompt("Q", "mine", &others, true);
        assert!(pp.contains("Maintain the requested academic structure."));
    }

    #[test]
    fn synthesis_prompt_joins_outputs() {
        let outs = map(&[("Alice", "x"), ("Bob", "y")]);
        let p = build_synthesis_prompt("Q", &outs, false);
        assert!(p.contains("[Alice]\nx"));
        assert!(p.contains("[Bob]\ny"));
        assert!(p.ends_with("Final Answer:"));
        assert!(build_synthesis_prompt("Q", &outs, true).contains("Ensure academic structure"));
    }

    #[test]
    fn system_prompt_uses_role_then_name_fallback() {
        assert!(build_system_prompt("Bot", Some("math")).contains("specialized in math"));
        assert!(build_system_prompt("Bot", None).contains("specialized in Bot"));
    }
}
