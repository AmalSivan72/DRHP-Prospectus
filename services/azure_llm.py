import os, json, time, logging
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

class AzureOpenAIClient:
    def __init__(self):
        self.key_vault_url = os.environ.get("KEY_VAULT_URL") or "https://KV-fs-to-autogen.vault.azure.net/"
        self._config = None
        self._client = None

    def _load_config_from_vault(self):
        if self._config is None:
            kv_url = os.environ.get("KEY_VAULT_URL") or self.key_vault_url
            try:
                credential = DefaultAzureCredential()
                client = SecretClient(vault_url=kv_url, credential=credential)
                cfg = {}
                try: cfg["api_key"] = client.get_secret("AzureLLMKey").value
                except: cfg["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY")
                try: cfg["api_base"] = client.get_secret("AzureOpenAiBase").value
                except: cfg["api_base"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
                try: cfg["model_version"] = client.get_secret("AzureOpenAiVersion").value
                except: cfg["model_version"] = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
                try: cfg["deployment"] = client.get_secret("AzureOpenAiDeployment").value
                except: cfg["deployment"] = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                self._config = cfg
            except Exception as e:
                logging.warning(f"Key Vault access failed: {e}, falling back to env vars")
                self._config = {
                    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                    "api_base": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    "model_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                    "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
                }
        return self._config

    def _get_client(self):
        if self._client is None:
            cfg = self._load_config_from_vault()
            self._client = AzureOpenAI(
                azure_endpoint=cfg["api_base"],
                api_key=cfg["api_key"],
                api_version=cfg["model_version"],
            )
        return self._client

    def evaluate(self, prompt: str, cache_path: str, max_retries: int = 3, backoff_factor: int = 2) -> dict:
        """
        Behaves like evaluate_with_gemini: send prompt, retry on errors, save cache, return parsed JSON.
        """
        client = self._get_client()
        deployment = self._config["deployment"]
        summary_filename = os.path.basename(cache_path)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()

                # Strip Markdown fencing
                if raw.startswith("```json"):
                    raw = raw[len("```json"):].strip()
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(raw)

                logging.info(f"Summary saved to: {summary_filename}")
                return json.loads(raw)

            except Exception as e:
                # Retry on rate limit or server errors
                status = getattr(e, "status_code", None)
                if status == 429 or (status and 500 <= status < 600):
                    wait = backoff_factor ** attempt
                    logging.warning(f"Azure OpenAI error {status}: {e}")
                    logging.warning(f"Retrying in {wait} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                else:
                    logging.error(f"Azure OpenAI call failed: {e}")
                    return {"answer": f"[API Error: {e}]", "reasoning_steps": [], "validation_steps": []}

        return {"answer": "[API Error: retries exhausted]", "reasoning_steps": [], "validation_steps": []}
