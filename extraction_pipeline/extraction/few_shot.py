"""
Six canonical few-shot examples covering heterogeneous document formats.
Each example is a (user_message, assistant_tool_call) pair for OpenAI messages.
"""

from __future__ import annotations

import json


def _assistant_tool_call(args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_example",
                "type": "function",
                "function": {
                    "name": "extract_document",
                    "arguments": json.dumps(args),
                },
            }
        ],
    }


# Format A — Narrative prose research paper
NARRATIVE_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "DOCUMENT ID: ex-001\n\n"
            "Quantum Entanglement in Distributed Systems\n\n"
            "Authors: Dr. Sarah Chen, Prof. Michael Torres\n"
            "Published: March 15, 2024 in Journal of Quantum Computing\n\n"
            "Abstract: This paper examines the role of quantum entanglement "
            "in modern distributed computing architectures [1][2].\n\n"
            "References:\n"
            "[1] Einstein A, Podolsky B, Rosen N. Can Quantum-Mechanical Description "
            "of Physical Reality Be Considered Complete? Phys Rev. 1935;47:777.\n"
            "[2] Bell JS. On the Einstein Podolsky Rosen paradox. Physics. 1964;1:195-200."
        ),
    },
    _assistant_tool_call({
        "document_id": "ex-001",
        "document_type": "research_paper",
        "document_type_other_detail": None,
        "title": "Quantum Entanglement in Distributed Systems",
        "author": "Dr. Sarah Chen, Prof. Michael Torres",
        "publication_date": "2024-03-15",
        "invoice_total": None,
        "currency": None,
        "citations": [
            "Einstein A, Podolsky B, Rosen N. Can Quantum-Mechanical Description of Physical Reality Be Considered Complete? Phys Rev. 1935;47:777.",
            "Bell JS. On the Einstein Podolsky Rosen paradox. Physics. 1964;1:195-200.",
        ],
        "confidence_scores": {
            "overall": 0.95,
            "title": 0.99,
            "author": 0.99,
            "publication_date": 0.97,
            "invoice_total": None,
            "citations": 0.95,
        },
    }),
]

# Format B — Structured table / key-value block
STRUCTURED_TABLE_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "DOCUMENT ID: ex-002\n\n"
            "Author     | Jane Smith\n"
            "Date       | 2023-03-17\n"
            "Title      | Market Risk Analysis Report\n"
            "Department | Risk Management\n"
            "Version    | 2.1\n\n"
            "Executive Summary: This report assesses market risk exposure across all trading desks."
        ),
    },
    _assistant_tool_call({
        "document_id": "ex-002",
        "document_type": "research_paper",
        "document_type_other_detail": None,
        "title": "Market Risk Analysis Report",
        "author": "Jane Smith",
        "publication_date": "2023-03-17",
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.98,
            "title": 0.99,
            "author": 0.99,
            "publication_date": 0.99,
            "invoice_total": None,
            "citations": None,
        },
    }),
]

# Format C — Inline citations [1][2]
INLINE_CITATIONS_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "DOCUMENT ID: ex-003\n\n"
            "Recent studies (Smith et al., 2022) [1] show a 34% reduction in latency "
            "when using edge caching. Other work by Lee and Park [2] confirms these "
            "findings in mobile contexts.\n\n"
            "This memo has no author or date listed.\n\n"
            "References:\n"
            "[1] Smith J, Jones A, Brown K. Edge Caching Performance. IEEE Trans. 2022.\n"
            "[2] Lee S, Park H. Mobile Edge Computing. ACM Conf. 2021."
        ),
    },
    _assistant_tool_call({
        "document_id": "ex-003",
        "document_type": "other",
        "document_type_other_detail": "technical memo",
        "title": None,
        "author": None,
        "publication_date": None,
        "invoice_total": None,
        "currency": None,
        "citations": [
            "Smith J, Jones A, Brown K. Edge Caching Performance. IEEE Trans. 2022.",
            "Lee S, Park H. Mobile Edge Computing. ACM Conf. 2021.",
        ],
        "confidence_scores": {
            "overall": 0.62,
            "title": None,
            "author": None,
            "publication_date": None,
            "invoice_total": None,
            "citations": 0.92,
        },
    }),
]

# Format D — Bibliography / Works Cited section
BIBLIOGRAPHY_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "DOCUMENT ID: ex-004\n\n"
            "Title: Supply Chain Disruption Analysis\n"
            "By: Robert Nguyen\n"
            "Date: January 2022\n\n"
            "Body text...\n\n"
            "Works Cited:\n"
            "Smith J. Market Risk. Oxford University Press. 2022.\n"
            "Chen L, Patel R. Global Supply Chains. MIT Press. 2020.\n"
            "Williams T. Disruption Theory. Harvard Bus Rev. 2019."
        ),
    },
    _assistant_tool_call({
        "document_id": "ex-004",
        "document_type": "research_paper",
        "document_type_other_detail": None,
        "title": "Supply Chain Disruption Analysis",
        "author": "Robert Nguyen",
        "publication_date": "2022-01-01",
        "invoice_total": None,
        "currency": None,
        "citations": [
            "Smith J. Market Risk. Oxford University Press. 2022.",
            "Chen L, Patel R. Global Supply Chains. MIT Press. 2020.",
            "Williams T. Disruption Theory. Harvard Bus Rev. 2019.",
        ],
        "confidence_scores": {
            "overall": 0.88,
            "title": 0.99,
            "author": 0.99,
            "publication_date": 0.80,
            "invoice_total": None,
            "citations": 0.95,
        },
    }),
]

# Format E — Invoice
INVOICE_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "DOCUMENT ID: ex-005\n\n"
            "INVOICE\n"
            "Invoice #: INV-2024-0892\n"
            "Date: 2024-07-15\n"
            "Bill To: Acme Corporation\n\n"
            "Services rendered: Cloud infrastructure consulting\n"
            "Subtotal: USD 4,175.00\n"
            "Tax (10%): USD 417.50\n"
            "Invoice Total: USD 4,592.50\n\n"
            "Payment due within 30 days."
        ),
    },
    _assistant_tool_call({
        "document_id": "ex-005",
        "document_type": "invoice",
        "document_type_other_detail": None,
        "title": None,
        "author": None,
        "publication_date": "2024-07-15",
        "invoice_total": 4592.50,
        "currency": "USD",
        "citations": None,
        "confidence_scores": {
            "overall": 0.97,
            "title": None,
            "author": None,
            "publication_date": 0.99,
            "invoice_total": 0.97,
            "citations": None,
        },
    }),
]

# Format F — Legal contract
CONTRACT_EXAMPLE = [
    {
        "role": "user",
        "content": (
            "DOCUMENT ID: ex-006\n\n"
            "SERVICE AGREEMENT\n\n"
            "This Service Agreement (\"Agreement\") is entered into as of 14 February 2025 "
            "by and between TechCorp Inc. (\"Provider\") and ClientCo Ltd. (\"Client\").\n\n"
            "WHEREAS, Provider desires to provide software development services;\n"
            "WHEREAS, Client desires to obtain such services;\n\n"
            "NOW, THEREFORE, the parties agree as follows:\n"
            "1. Services. Provider shall deliver the software as described in Exhibit A.\n"
            "2. Term. This Agreement commences on the Effective Date and continues for 12 months."
        ),
    },
    _assistant_tool_call({
        "document_id": "ex-006",
        "document_type": "contract",
        "document_type_other_detail": None,
        "title": "SERVICE AGREEMENT",
        "author": None,
        "publication_date": "2025-02-14",
        "invoice_total": None,
        "currency": None,
        "citations": None,
        "confidence_scores": {
            "overall": 0.93,
            "title": 0.95,
            "author": None,
            "publication_date": 0.93,
            "invoice_total": None,
            "citations": None,
        },
    }),
]

ALL_EXAMPLES = {
    "narrative": NARRATIVE_EXAMPLE,
    "structured_table": STRUCTURED_TABLE_EXAMPLE,
    "inline_citations": INLINE_CITATIONS_EXAMPLE,
    "bibliography": BIBLIOGRAPHY_EXAMPLE,
    "invoice": INVOICE_EXAMPLE,
    "contract": CONTRACT_EXAMPLE,
}

_KEYWORD_MAP = {
    "invoice": ["invoice", "total due", "bill to", "amount due", "$", "usd", "eur", "gbp"],
    "contract": ["agreement", "whereas", "hereby", "parties", "effective date", "term"],
    "inline_citations": ["[1]", "[2]", "[3]", "et al.", "op. cit."],
    "bibliography": ["works cited", "references:", "bibliography", "press.", "journal"],
    "narrative": ["abstract", "introduction", "conclusion", "published", "journal of"],
    "structured_table": ["|", "author |", "date |", "title |", "version |"],
}


class FewShotLibrary:
    def get_examples(self, document_hint: str, n: int = 2) -> list[dict]:
        """Select n most relevant few-shot examples based on keyword matching."""
        hint_lower = document_hint.lower()
        scores: dict[str, int] = {k: 0 for k in _KEYWORD_MAP}
        for fmt, keywords in _KEYWORD_MAP.items():
            for kw in keywords:
                if kw in hint_lower:
                    scores[fmt] += 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected: list[dict] = []
        for fmt, _score in ranked[:n]:
            selected.extend(ALL_EXAMPLES[fmt])
        return selected
