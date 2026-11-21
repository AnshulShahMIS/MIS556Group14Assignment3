import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parents[1]
BRANDS = ('AmazonHelp', 'AppleSupport', 'Uber_Support', 'SpotifyCares', 'Delta')
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


@dataclass
class Settings:
    provider: str = 'openai'
    model: str = 'gpt-4.1-mini'
    api_key: str = field(default='',repr=False)
    runtime: Path = APP_ROOT / '.runtime'
    data_dir: Path = APP_ROOT / 'data'
    input_price: float | None = 0.40
    output_price: float | None = 1.60
    evaluation_root: Path | None = None

    @classmethod
    def from_env(cls):
        load_dotenv(os.getenv('SUPPORT_ENV_FILE', APP_ROOT / '.env'))
        provider = os.getenv('LLM_PROVIDER', 'openai')
        model = os.getenv('LLM_MODEL', 'gpt-4.1-mini' if provider == 'openai' else 'claude-sonnet-4-6')
        default_prices = (0.40, 1.60) if provider == 'openai' and model == 'gpt-4.1-mini' else (None, None)
        return cls(provider=provider, model=model,
                   api_key=os.getenv('OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY', ''),
                   runtime=Path(os.getenv('SUPPORT_RUNTIME_DIR', APP_ROOT / '.runtime')),
                   input_price=float(os.environ['INPUT_PRICE_PER_MILLION']) if os.getenv('INPUT_PRICE_PER_MILLION') else default_prices[0],
                   output_price=float(os.environ['OUTPUT_PRICE_PER_MILLION']) if os.getenv('OUTPUT_PRICE_PER_MILLION') else default_prices[1])
