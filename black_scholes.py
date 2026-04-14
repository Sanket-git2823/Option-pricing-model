
import math
import argparse
import numpy as np
from scipy.stats import norm

def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """
    Compute d1 and d2 terms used throughout Black-Scholes.

    Parameters
    ----------
    S     : float  – Current spot price
    K     : float  – Strike price
    T     : float  – Time to expiry in years
    r     : float  – Risk-free rate (annualised, decimal)
    sigma : float  – Volatility (annualised, decimal)
    q     : float  – Continuous dividend yield (decimal), default 0

    Returns
    -------
    (d1, d2) : tuple[float, float]
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European call price."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European put price (via put-call parity)."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> dict:
    """
    Compute all first-order and second-order Greeks for calls and puts.

    Returns a dict with keys:
      delta_call, delta_put,
      gamma,
      theta_call, theta_put   (per calendar day)
      vega                    (per 1 % move in vol)
      rho_call, rho_put       (per 1 % move in rate)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    Nd1   = norm.cdf(d1)
    Nnd1  = norm.cdf(-d1)
    Nd2   = norm.cdf(d2)
    Nnd2  = norm.cdf(-d2)
    nd1   = norm.pdf(d1)        # standard normal PDF at d1

    eq_T  = math.exp(-q * T)
    er_T  = math.exp(-r * T)

    delta_call = eq_T * Nd1
    delta_put  = -eq_T * Nnd1

    gamma = eq_T * nd1 / (S * sigma * math.sqrt(T))

    theta_call = (
        -S * eq_T * nd1 * sigma / (2 * math.sqrt(T))
        - r * K * er_T * Nd2
        + q * S * eq_T * Nd1
    ) / 365

    theta_put = (
        -S * eq_T * nd1 * sigma / (2 * math.sqrt(T))
        + r * K * er_T * Nnd2
        - q * S * eq_T * Nnd1
    ) / 365

    vega     = S * eq_T * nd1 * math.sqrt(T) * 0.01   # per 1 % vol
    rho_call = K * T * er_T * Nd2  * 0.01             # per 1 % rate
    rho_put  = -K * T * er_T * Nnd2 * 0.01

    return {
        "d1": d1, "d2": d2,
        "N_d1": Nd1, "N_d2": Nd2,
        "delta_call": delta_call, "delta_put": delta_put,
        "gamma": gamma,
        "theta_call": theta_call, "theta_put": theta_put,
        "vega": vega,
        "rho_call": rho_call, "rho_put": rho_put,
    }


def price_all(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> dict:
    """
    Return call price, put price, intrinsic values, and all Greeks in one dict.
    """
    cp = call_price(S, K, T, r, sigma, q)
    pp = put_price(S, K, T, r, sigma, q)
    g  = greeks(S, K, T, r, sigma, q)

    intrinsic_call = max(0.0, S - K)
    intrinsic_put  = max(0.0, K - S)

    return {
        "call_price":       round(cp, 6),
        "put_price":        round(pp, 6),
        "intrinsic_call":   round(intrinsic_call, 6),
        "intrinsic_put":    round(intrinsic_put,  6),
        "time_value_call":  round(cp - intrinsic_call, 6),
        "time_value_put":   round(pp - intrinsic_put,  6),
        **{k: round(v, 6) for k, v in g.items()},
    }


def payoff_curve(S: float, K: float, T: float, r: float, sigma: float,
                 q: float = 0.0, n_points: int = 80):
    """
    Return spot-vs-payoff and spot-vs-option-price arrays for plotting.
    Spot range: [0.5 S, 1.5 S]
    """
    spots = np.linspace(0.5 * S, 1.5 * S, n_points).tolist()

    cp = call_price(S, K, T, r, sigma, q)
    pp = put_price(S, K, T, r, sigma, q)

    call_payoff, put_payoff = [], []
    call_prices, put_prices = [], []

    for s in spots:
        call_payoff.append(round(max(0, s - K) - cp, 4))
        put_payoff.append(round(max(0, K - s) - pp, 4))
        call_prices.append(round(call_price(s, K, T, r, sigma, q), 4))
        put_prices.append(round(put_price(s, K, T, r, sigma, q), 4))

    return {
        "spots":        [round(s, 2) for s in spots],
        "call_payoff":  call_payoff,
        "put_payoff":   put_payoff,
        "call_prices":  call_prices,
        "put_prices":   put_prices,
    }

def create_app():
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    def parse_params(req):
        """Extract and validate pricing parameters from query string or JSON body."""
        data = req.get_json(silent=True) or req.args
        S     = float(data.get("S",     100))
        K     = float(data.get("K",     105))
        T     = float(data.get("T",     0.5))
        r     = float(data.get("r",     0.05))
        sigma = float(data.get("sigma", 0.20))
        q     = float(data.get("q",     0.0))
        assert S > 0 and K > 0 and T > 0 and sigma > 0, "Invalid parameter values"
        return S, K, T, r, sigma, q

    @app.route("/price", methods=["GET", "POST"])
    def price():
        """Return full pricing result including Greeks."""
        try:
            S, K, T, r, sigma, q = parse_params(request)
            result = price_all(S, K, T, r, sigma, q)
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/curves", methods=["GET", "POST"])
    def curves():
        """Return chart curve data."""
        try:
            S, K, T, r, sigma, q = parse_params(request)
            result = payoff_curve(S, K, T, r, sigma, q)
            return jsonify({"ok": True, "data": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


def demo():
    """Print a sample Black-Scholes pricing to stdout."""
    params = dict(S=100, K=105, T=0.5, r=0.05, sigma=0.20, q=0.0)
    result = price_all(**params)

    print("\n── Black-Scholes Demo ──────────────────────────")
    print(f"  S={params['S']}  K={params['K']}  T={params['T']}y  "
          f"r={params['r']*100:.1f}%  σ={params['sigma']*100:.1f}%  q={params['q']*100:.1f}%")
    print(f"\n  Call price : ${result['call_price']:.4f}")
    print(f"  Put price  : ${result['put_price']:.4f}")
    print(f"\n  Greeks (call / put)")
    print(f"    Delta  : {result['delta_call']:.4f} / {result['delta_put']:.4f}")
    print(f"    Gamma  : {result['gamma']:.5f}")
    print(f"    Theta  : {result['theta_call']:.4f} / {result['theta_put']:.4f}  (per day)")
    print(f"    Vega   : {result['vega']:.4f}  (per 1% vol)")
    print(f"    Rho    : {result['rho_call']:.4f} / {result['rho_put']:.4f}  (per 1% rate)")
    print(f"\n  d1={result['d1']:.4f}  d2={result['d2']:.4f}")
    print(f"  N(d1)={result['N_d1']:.4f}  N(d2)={result['N_d2']:.4f}")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-Scholes pricer")
    parser.add_argument("--demo", action="store_true", help="Print demo output and exit")
    parser.add_argument("--port", type=int, default=5000, help="Flask port (default 5000)")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        print(f"Starting Black-Scholes API on http://localhost:{args.port}")
        print("Endpoints:  GET /price?S=100&K=105&T=0.5&r=0.05&sigma=0.2")
        print("            GET /curves?S=100&K=105&T=0.5&r=0.05&sigma=0.2")
        app = create_app()
        app.run(port=args.port, debug=False)
