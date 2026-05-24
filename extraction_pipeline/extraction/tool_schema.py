"""
Single source of truth for the OpenAI extraction tool definition.
strict=True forces explicit null for absent fields — primary anti-hallucination mechanism.
"""

EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "document_id",
        "document_type",
        "document_type_other_detail",
        "title",
        "author",
        "publication_date",
        "invoice_total",
        "currency",
        "citations",
        "confidence_scores",
    ],
    "additionalProperties": False,
    "properties": {
        "document_id": {"type": "string"},
        "document_type": {
            "type": "string",
            "enum": ["invoice", "contract", "research_paper", "resume", "medical_record", "other"],
        },
        "document_type_other_detail": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        },
        "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "author": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "publication_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "invoice_total": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "currency": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "citations": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        },
        "confidence_scores": {
            "type": "object",
            "required": ["overall"],
            "additionalProperties": False,
            "properties": {
                "overall": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "title": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        {"type": "null"},
                    ]
                },
                "author": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        {"type": "null"},
                    ]
                },
                "publication_date": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        {"type": "null"},
                    ]
                },
                "invoice_total": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        {"type": "null"},
                    ]
                },
                "citations": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        {"type": "null"},
                    ]
                },
            },
        },
    },
}

EXTRACTION_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "extract_document",
        "description": (
            "Extract structured metadata from a document. "
            "Use null for any field not explicitly present in the source document. "
            "Never infer or fabricate values."
        ),
        "strict": True,
        "parameters": EXTRACTION_JSON_SCHEMA,
    },
}
