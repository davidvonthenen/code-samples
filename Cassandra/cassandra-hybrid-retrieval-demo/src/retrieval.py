"""Retrieval lanes shared by the CLI, the evaluator, and the presenter.

Three ways to answer the same question against one Cassandra table:

1. `keyword_search`  — SAI term matching on the `keywords` set
2. `vector_search`   — JVector ANN over the embedding column
3. `filtered_ann`    — SAI predicates and ANN ordering in one CQL statement
"""

from __future__ import annotations

import string
from dataclasses import asdict, dataclass, replace
from weakref import WeakKeyDictionary

from common import TABLE, embed_query

DEFAULT_QUERIES = [
    "What is recall 24V-330?",
    "Is there a recall on the tailgate?",
    "How much can the Summit 1500 tow?",
]

MAX_LIMIT = 10

# Token the presenter flags in retrieved chunks: the model year being asked about.
HIGHLIGHT_TOKEN = "2024"

SELECT_COLUMNS = "id, title, category, body, model, model_year"

# Bare nouns worth grounding on. Identifiers and numbers are matched by shape.
KNOWN_TERMS = frozenset(
    {
        "summit",
        "tow",
        "towing",
        "payload",
        "recall",
        "tailgate",
        "hitch",
        "warranty",
        "oil",
        "tire",
    }
)

# Preparing costs a round trip, so keep one statement per session and shape
# without keeping the session itself alive.
_STATEMENTS: WeakKeyDictionary = WeakKeyDictionary()


@dataclass
class Hit:
    id: str
    title: str
    category: str
    body: str
    model: str | None = None
    model_year: int | None = None
    score: float | None = None
    # How many query tokens the document matched. Set by the keyword lane only,
    # where it is the app's ranking signal rather than a relevance score.
    matched: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    tokens: list[str]
    keyword: str | None
    category: str | None
    model: str | None
    model_year: int | None
    keyword_hits: list[Hit]
    vector_hits: list[Hit]
    filtered_hits: list[Hit]
    filtered_cql: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "tokens": self.tokens,
            "keyword": self.keyword,
            "category": self.category,
            "model": self.model,
            "model_year": self.model_year,
            "keyword_hits": [hit.to_dict() for hit in self.keyword_hits],
            "vector_hits": [hit.to_dict() for hit in self.vector_hits],
            "filtered_hits": [hit.to_dict() for hit in self.filtered_hits],
            "filtered_cql": self.filtered_cql,
        }


def _prepared(session, key: str, cql: str):
    statements = _STATEMENTS.setdefault(session, {})
    statement = statements.get(key)
    if statement is None:
        statement = session.prepare(cql)
        statements[key] = statement
    return statement


def _row_to_hit(row, score: float | None = None, matched: int | None = None) -> Hit:
    return Hit(
        id=row.id,
        title=row.title,
        category=row.category,
        body=row.body,
        model=getattr(row, "model", None),
        model_year=getattr(row, "model_year", None),
        score=score,
        matched=matched,
    )


def _bounded(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def extract_keyword_tokens(query: str) -> list[str]:
    """Pull the exact tokens worth grounding on, lowercased to match the corpus.

    Identifiers (`24v-330`) and short numbers (`2024`, `1500`) are recognized by
    shape; everything else has to be a known domain term.
    """
    tokens: list[str] = []
    for raw in query.replace("/", " ").split():
        token = raw.strip(string.punctuation).lower()
        has_digit = any(character.isdigit() for character in token)
        is_number = token.isdigit() and 3 <= len(token) <= 4
        is_identifier = has_digit and not token.isdigit()
        if is_number or is_identifier or token in KNOWN_TERMS:
            if token not in tokens:
                tokens.append(token)
    return tokens


def selective_token(query: str) -> str | None:
    """Return the one exact token that narrows the corpus, if the query has one.

    `keywords CONTAINS 'recall'` matches all twelve recalls and buys nothing;
    `keywords CONTAINS '24v-330'` matches one. Choosing the identifier-shaped
    token is the difference between a hybrid query that fixes the ranking and
    one that reorders the same wrong candidates. Real systems put their own ID
    patterns here — SKUs, error codes, part numbers.
    """
    for token in extract_keyword_tokens(query):
        if any(character.isdigit() for character in token) and not token.isdigit():
            return token
    return None


def infer_category(query: str) -> str | None:
    """Map a question to one of the corpus categories, or None if ambiguous."""
    lowered = query.lower()
    if "recall" in lowered or "tailgate" in lowered:
        return "recalls"
    if any(word in lowered for word in ("oil", "tire", "maintenance", "service")):
        return "maintenance"
    if any(word in lowered for word in ("tow", "towing", "trailer", "hitch")):
        return "towing"
    if "warranty" in lowered or "coverage" in lowered:
        return "warranty"
    if any(word in lowered for word in ("payload", "spec", "cargo")):
        return "specs"
    return None


def keyword_search(session, query: str, limit: int = 5) -> list[Hit]:
    """Match exact tokens with SAI, then rank by how many tokens each doc matched.

    SAI returns matching rows, not relevance scores. The overlap count is the
    app's own ranking step, so documents matching the same number of tokens are
    in arbitrary order. Read that tail as a set of matches, not a ranking.
    """
    statement = _prepared(
        session,
        "keyword",
        f"SELECT {SELECT_COLUMNS} FROM {TABLE} WHERE keywords CONTAINS ?",
    )
    matched: dict[str, int] = {}
    documents: dict[str, Hit] = {}
    order: dict[str, int] = {}
    for token in extract_keyword_tokens(query):
        for row in session.execute(statement, (token,)):
            matched[row.id] = matched.get(row.id, 0) + 1
            documents.setdefault(row.id, _row_to_hit(row))
            order.setdefault(row.id, len(order))

    ranked = sorted(matched, key=lambda doc_id: (-matched[doc_id], order[doc_id]))
    return [
        replace(documents[doc_id], matched=matched[doc_id])
        for doc_id in ranked[: _bounded(limit)]
    ]


def vector_search(session, query: str, limit: int = 5) -> list[Hit]:
    """Return semantic neighbors ordered by Cassandra ANN."""
    statement = _prepared(
        session,
        "ann",
        f"""
        SELECT {SELECT_COLUMNS},
               similarity_cosine(embedding, ?) AS score
        FROM {TABLE}
        ORDER BY embedding ANN OF ?
        LIMIT ?
        """,
    )
    embedding = embed_query(query)
    rows = session.execute(statement, (embedding, embedding, _bounded(limit)))
    return [
        _row_to_hit(row, score=float(row.score) if row.score is not None else None)
        for row in rows
    ]


def _filter_clauses(
    keyword: str | None = None,
    category: str | None = None,
    model: str | None = None,
    model_year: int | None = None,
) -> list[tuple[str, object]]:
    """Return the active SAI predicates in a stable order, one `?` each."""
    candidates = (
        ("keywords CONTAINS ?", keyword),
        ("category = ?", category),
        ("model = ?", model),
        ("model_year = ?", model_year),
    )
    return [(clause, value) for clause, value in candidates if value is not None]


def _ann_cql(where: str, limit: object = "?") -> str:
    return (
        f"SELECT {SELECT_COLUMNS},\n"
        "       similarity_cosine(embedding, ?) AS score\n"
        f"FROM {TABLE}\n"
        f"WHERE {where}\n"
        "ORDER BY embedding ANN OF ?\n"
        f"LIMIT {limit}"
    )


def _cql_literal(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def filtered_where(
    keyword: str | None = None,
    category: str | None = None,
    model: str | None = None,
    model_year: int | None = None,
) -> str | None:
    """Render the WHERE clause `filtered_ann` runs, with values inlined."""
    clauses = _filter_clauses(keyword, category, model, model_year)
    if not clauses:
        return None
    return " AND ".join(
        clause.replace("?", _cql_literal(value)) for clause, value in clauses
    )


def filtered_ann_cql(
    keyword: str | None = None,
    category: str | None = None,
    model: str | None = None,
    model_year: int | None = None,
    limit: int = 5,
) -> str | None:
    """Render the statement `filtered_ann` runs, for display next to its hits.

    Filters and the limit are inlined; the query vector stays a bind marker
    because a 384-dimensional literal tells a reader nothing.
    """
    where = filtered_where(keyword, category, model, model_year)
    return _ann_cql(where, _bounded(limit)) + ";" if where else None


def filtered_ann(
    session,
    query: str,
    keyword: str | None = None,
    category: str | None = None,
    model: str | None = None,
    model_year: int | None = None,
    limit: int = 5,
) -> list[Hit]:
    """Hybrid: SAI equality and term filters, then ANN ordering.

    One statement, one index-supported scan. This is the lane that actually
    fixes an identifier lookup, and the reason it does is the `WHERE` clause,
    not the vector.
    """
    clauses = _filter_clauses(keyword, category, model, model_year)
    if not clauses:
        return vector_search(session, query, limit)

    where = " AND ".join(clause for clause, _ in clauses)
    statement = _prepared(session, f"filtered:{where}", _ann_cql(where))
    embedding = embed_query(query)
    values = [value for _, value in clauses]
    rows = session.execute(
        statement, (embedding, *values, embedding, _bounded(limit))
    )
    return [
        _row_to_hit(row, score=float(row.score) if row.score is not None else None)
        for row in rows
    ]


def retrieve_all(
    session,
    query: str,
    limit: int = 5,
    model: str | None = None,
    model_year: int | None = None,
) -> RetrievalResult:
    """Run every lane once and return a presenter-neutral result."""
    keyword = selective_token(query)
    category = infer_category(query)
    keyword_hits = keyword_search(session, query, limit)
    vector_hits = vector_search(session, query, limit)
    return RetrievalResult(
        query=query,
        tokens=extract_keyword_tokens(query),
        keyword=keyword,
        category=category,
        model=model,
        model_year=model_year,
        keyword_hits=keyword_hits,
        vector_hits=vector_hits,
        filtered_hits=filtered_ann(
            session,
            query,
            keyword=keyword,
            category=category,
            model=model,
            model_year=model_year,
            limit=limit,
        )
        if _filter_clauses(keyword, category, model, model_year)
        else [],
        filtered_cql=filtered_ann_cql(keyword, category, model, model_year, limit),
    )
