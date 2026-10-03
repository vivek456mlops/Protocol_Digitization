"""Public, provenance-aware scientific context from Europe PMC open-access records."""

from typing import Dict, List, Optional

import httpx


class EuropePMCProvider:
    def __init__(self, base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest") -> None:
        self.base_url = base_url.rstrip("/")

    def search(self, question: str, limit: int = 3) -> List[Dict]:
        query = f'OPEN_ACCESS:Y AND ({question})'
        response = httpx.get(
            f"{self.base_url}/search",
            params={"query": query, "format": "json", "resultType": "core", "pageSize": limit},
            timeout=8,
        )
        response.raise_for_status()
        results = []
        for item in response.json().get("resultList", {}).get("result", []):
            abstract = item.get("abstractText")
            if not abstract:
                continue
            results.append({
                "id": item.get("pmcid") or item.get("id"),
                "title": item.get("title", "Europe PMC record"),
                "text": abstract,
                "source": "Europe PMC",
                "kind": "public_open_access",
                "license": item.get("license") or "Open-access status returned by Europe PMC; verify article license before redistribution.",
                "url": f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id')}",
            })
        return results


def configured_public_provider() -> Optional[EuropePMCProvider]:
    import os

    if os.getenv("PUBLIC_KNOWLEDGE_PROVIDER", "none").lower() == "europepmc":
        return EuropePMCProvider(os.getenv("EUROPE_PMC_URL", "https://www.ebi.ac.uk/europepmc/webservices/rest"))
    return None
