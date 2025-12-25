# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binance BTC futures auto-trading agent with a hybrid architecture: rule-based signals + LLM context filtering.

**Target**: Scalping + Swing trading automation

## Tech Stack

- Python 3.11+, asyncio/aiohttp
- Redis (real-time cache), SQLite (history)
- Binance Futures API, Claude API (Anthropic)

## Commands

```bash
# Development
make dev              # Run dev server
make test             # Run all tests
make lint             # Lint check
pytest tests/signals/ # Run specific test directory

# Backtest
make backtest         # Full backtest
make backtest-scalp   # Scalp only
make backtest-swing   # Swing only

# Deployment
make deploy           # Deploy to VPS
```

## Architecture

```
Data Collectors → Signal Generator → LLM Filter → Risk Manager → Executor
     │                  │                │              │            │
     ▼                  ▼                ▼              ▼            ▼
  Redis           Rule-based         Claude API    Mandatory     Binance
  Cache           signals            judgment      rules         Futures
```

### Data Flow

1. **Collectors** continuously feed market data (WebSocket), macro data, news into Redis
2. **Signal generators** check conditions every tick/candle and emit `Signal` objects
3. **LLM filter** evaluates signals with full market context, returns `LLMDecision`
4. **Risk manager** applies mandatory rules (daily loss limit, cooldowns, event blackout)
5. **Executor** places orders via Binance Futures API

### Key Design Decisions

- **Single agent** (not multi-agent) - MVP simplicity
- **LLM as filter, not generator** - rules for speed, LLM for context judgment
- **Scalp + Swing integrated** - prevents position conflicts (same direction only)

## Core Interfaces

```python
@dataclass
class Signal:
    type: Literal["SCALP", "SWING"]
    direction: Literal["LONG", "SHORT"]
    confidence: float  # 0.0 ~ 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime

@dataclass
class LLMDecision:
    execute: bool
    confidence: float
    adjusted_size: float  # 0.0 ~ 1.0
    reason: str

class BaseCollector(ABC):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def subscribe(self, channels: list[str]) -> None: ...
```

## Coding Conventions

- Type hints required on all functions
- Google-style docstrings
- Async code uses asyncio (not threading)
- Environment variables in `.env` file
- All signal logic must have unit tests

## Required Environment Variables

```bash
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
ANTHROPIC_API_KEY=
FRED_API_KEY=
REDIS_URL=redis://localhost:6379
```

## Subagents

Automatically invoked based on task context:

- `data-collector` - WebSocket/REST/macro/news data pipelines
- `signal-generator` - Scalp/swing signal logic, technical indicators
- `risk-manager` - Position sizing, protection rules, exposure limits
- `backtester` - Backtest engine, performance analysis
- `code-reviewer` - Code quality, security review
- `llm-brain` - Claude API integration, prompt engineering

## Slash Commands

- `/setup` - Initial project setup (dependencies, env, directories)
- `/status` - Check system status (Redis, API connections, env)
- `/backtest` - Run backtest with performance report
- `/add-signal <name> <scalp|swing>` - Add new signal from template

## Key Documentation

- `docs/architecture.md` - Component diagram, data flow, module details
- `docs/signals.md` - Signal definitions, trigger conditions, combinations
- `docs/risk-rules.md` - Position limits, SL/TP, protection rules
- `docs/design.md` - Overall design decisions, data sources, implementation phases
