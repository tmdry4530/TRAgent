# Risk Management Module

Comprehensive risk management implementation for the trading agent.

## Overview

The risk module provides two main components:

1. **PositionCalculator** - Calculate optimal position sizes based on account balance and risk parameters
2. **RiskManager** - Enforce protection rules and prevent dangerous trades

## Installation

The module is already included in the project. No additional installation needed.

## Quick Start

```python
import yaml
from src.risk import PositionCalculator, RiskManager
from src.signals.base import Signal
from datetime import datetime

# Load configuration
with open("config/trading.yaml") as f:
    config = yaml.safe_load(f)

# Initialize components
calculator = PositionCalculator(config)
risk_manager = RiskManager(config)

# Create a signal
signal = Signal(
    type="SCALP",
    direction="LONG",
    confidence=0.85,
    entry_price=50000.0,
    stop_loss=49250.0,
    take_profit=52500.0,
    reason="Volume breakout",
    timestamp=datetime.now(),
)

# Check risk
account_state = {
    "balance": 10000.0,
    "positions": [],
    "llm_confidence": 0.75,
}

risk_check = risk_manager.check(signal, account_state)

if risk_check.approved:
    # Calculate position size
    position = calculator.calculate(
        signal=signal,
        account_balance=10000.0,
        current_price=50000.0,
    )

    print(f"Position: {position.size_usd} USD")
    print(f"Leverage: {position.leverage}x")
    print(f"Risk: ${position.risk_amount}")
else:
    print(f"Trade blocked: {risk_check.reason}")
```

## Components

### PositionCalculator

Calculates position sizes using a Kelly-like approach:

```python
calculator = PositionCalculator(config)

position = calculator.calculate(
    signal=signal,
    account_balance=10000.0,
    current_price=50000.0,
    existing_positions=[],
)

# Returns PositionSize with:
# - size_usd: Position size in USD
# - size_btc: Position size in BTC
# - leverage: Applied leverage (1-40x for scalp, 1-10x for swing)
# - risk_amount: Amount at risk
```

**Position Limits:**
- SCALP: Max 30% of account, 20-40x leverage
- SWING: Max 50% of account, 5-10x leverage
- Total exposure: Max 80% of account

### RiskManager

Enforces protection rules:

```python
risk_manager = RiskManager(config)

# Perform risk check
result = risk_manager.check(signal, account_state)

if result.approved:
    # Trade approved
    final_size = position.size_usd * result.adjusted_size
else:
    # Trade blocked
    print(result.reason)

# Record trade result
from src.risk import TradeResult

trade_result = TradeResult(
    timestamp=datetime.now(),
    pnl=-150.0,
    pnl_pct=-1.5,
    signal_type="SCALP",
    direction="LONG",
)

risk_manager.record_trade_result(trade_result)
```

## Protection Rules

### 1. Daily Loss Limit (10%)

Trading stops when daily loss reaches 10% of account balance.

```python
# Automatically enforced
if abs(daily_pnl) >= account_balance * 0.10:
    # Block all new trades until next day
```

### 2. Consecutive Loss Cooldown

After 3 consecutive losses, trigger 1-hour cooldown.

```python
# Automatically enforced
if consecutive_losses >= 3:
    cooldown_until = now + timedelta(hours=1)
    # No new trades during cooldown
```

### 3. Event Blackout

No new trades 30 minutes before major economic events (FOMC, CPI, etc.).

```python
# Update events
risk_manager.update_events([
    {
        "name": "FOMC",
        "time": event_datetime.isoformat(),
    }
])

# Automatically checked
if is_event_blackout():
    # Block new trades
```

### 4. LLM Confidence Threshold

Position size reduced to 50% if LLM confidence < 0.6.

```python
account_state = {
    "balance": 10000.0,
    "positions": [],
    "llm_confidence": 0.55,  # Below threshold
}

result = risk_manager.check(signal, account_state)
# result.adjusted_size == 0.5
```

### 5. Position Conflicts

Only same-direction positions allowed:
- SCALP LONG + SWING LONG ✓
- SCALP SHORT + SWING SHORT ✓
- SCALP LONG + SWING SHORT ✗
- Two SCALP positions ✗

## Stop Loss & Take Profit

| Type | Stop Loss | Take Profit | Trailing Stop |
|------|-----------|-------------|---------------|
| SCALP | 1.5% | 3-5% | None |
| SWING | 5% | 15-25% | 7% from peak |

All signals must have valid levels:
- LONG: `stop_loss < entry_price < take_profit`
- SHORT: `take_profit < entry_price < stop_loss`

## Statistics

Track risk metrics:

```python
stats = risk_manager.get_stats()

print(f"Daily PnL: ${stats['daily_pnl']:.2f}")
print(f"Consecutive Losses: {stats['consecutive_losses']}")
print(f"Cooldown Until: {stats['cooldown_until']}")
print(f"Trades Today: {stats['total_trades_today']}")
print(f"Event Blackout: {stats['event_blackout']}")
```

## Configuration

Edit `config/trading.yaml`:

```yaml
scalp:
  max_leverage: 40
  position_size: 0.3  # 30% of account
  stop_loss: 0.015    # 1.5%
  take_profit: 0.05   # 5%

swing:
  max_leverage: 10
  position_size: 0.5  # 50% of account
  stop_loss: 0.05     # 5%
  take_profit: 0.25   # 25%
  trailing_stop: 0.07 # 7%

risk:
  daily_loss_limit: 0.1              # 10%
  consecutive_loss_cooldown: 3        # 3 losses
  event_blackout_minutes: 30         # 30 minutes
  llm_confidence_threshold: 0.6      # 60%
```

## Testing

Run comprehensive tests:

```bash
pytest tests/risk/ -v
```

Test coverage:
- Position size calculation
- Exposure limits
- Daily loss limit enforcement
- Cooldown triggers
- Event blackout
- Position conflicts
- LLM confidence adjustments

## Examples

See `examples/risk_example.py` for complete usage examples:

```bash
export PYTHONPATH="D:\Develop\Agent\trading-agent"
python examples/risk_example.py
```

## Files

- `src/risk/calculator.py` - Position size calculator
- `src/risk/manager.py` - Risk manager with protection rules
- `src/risk/__init__.py` - Module exports
- `tests/risk/` - Unit tests
- `examples/risk_example.py` - Usage examples
- `docs/risk-rules.md` - Detailed rules documentation (Korean)

## API Reference

### PositionSize

```python
@dataclass
class PositionSize:
    size_usd: float      # Position size in USD
    size_btc: float      # Position size in BTC
    leverage: int        # Applied leverage
    risk_amount: float   # Amount at risk
```

### RiskCheckResult

```python
@dataclass
class RiskCheckResult:
    approved: bool                   # Whether trade is approved
    reason: str | None              # Rejection reason if not approved
    adjusted_size: float            # Position size multiplier (0.0-1.0)
    cooldown_until: datetime | None # Cooldown expiration if applicable
    warnings: list[str]             # Warning messages
```

### TradeResult

```python
@dataclass
class TradeResult:
    timestamp: datetime  # Trade completion time
    pnl: float          # Profit/loss amount
    pnl_pct: float      # Profit/loss percentage
    signal_type: str    # SCALP or SWING
    direction: str      # LONG or SHORT
```

## Integration

The risk module is designed to integrate with:

1. **Signal Generators** - Validate signals before execution
2. **LLM Filter** - Adjust position based on LLM confidence
3. **Executor** - Calculate final position sizes
4. **Backtester** - Simulate realistic risk management

Example integration:

```python
# Signal Generator
signal = generator.generate(market_state)

# LLM Filter
llm_decision = await llm_filter.evaluate(signal, context)

# Risk Check
account_state = {
    "balance": account.balance,
    "positions": account.positions,
    "llm_confidence": llm_decision.confidence,
}
risk_check = risk_manager.check(signal, account_state)

if risk_check.approved:
    # Calculate Position
    position = calculator.calculate(
        signal=signal,
        account_balance=account.balance,
        current_price=market_state.price,
        existing_positions=account.positions,
    )

    # Apply adjustments
    final_size = position.size_usd * risk_check.adjusted_size

    # Execute
    await executor.place_order(
        direction=signal.direction,
        size=final_size,
        leverage=position.leverage,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
    )
```

## License

Part of the trading-agent project.
