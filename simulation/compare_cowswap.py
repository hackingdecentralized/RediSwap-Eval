import json
import os
import pandas as pd
import time


from base import *
from utils import *


# Destination path
FIGURES_PATH = os.path.join(PARENT_DIR, "figures")
COWSWAP_PATH = os.path.join(DATA_PATH, "cowswap")

  
# Constants
FEES = [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
SAMPLES = [3, 10, 20]


# Load cowswap data
def load_cowswap(month_str):
    # Load the corresponding CSV file
    month_path = os.path.join(COWSWAP_PATH, f"{month_str}.csv.zip")
    df = pd.read_csv(month_path, compression="zip")
    return df


def simulate():
    date_ranges = pd.date_range(start="2023-09-01", end="2024-08-31", freq="MS")
    results = {}
    prices = []

    for fee in FEES:
        results[fee] = {
            "gaussian": {k: [] for k in SAMPLES},
            "pareto": {k: [] for k in SAMPLES},
        }

    for month in date_ranges:
        month_str = month.strftime("%Y%m")
        print(f"Processing {month_str}...")
        binance_prices = load_monthly_prices(month_str)
        cowswap_data_df = load_cowswap(month_str)
        cowswap_data_df["block_timestamp"] = pd.to_datetime(cowswap_data_df["block_timestamp"])
        print(f"Data loaded for {month_str}, simulation starts...")

        for _, row in cowswap_data_df.iterrows():
            token_input_addr = row["token_sold_address"]
            token_output_addr = row["token_bought_address"]
            user_request_time = row["block_timestamp"].timestamp()
            
            if token_input_addr not in ADDR_TO_COINS or token_output_addr not in ADDR_TO_COINS:
                continue

            token_input_amount = settled_amount_in = float(row["token_sold_amount_raw"])
            token_output_amount = settled_amount_out = float(row["token_bought_amount_raw"])

            token_input = ADDR_TO_COINS[token_input_addr][0]
            token_output = ADDR_TO_COINS[token_output_addr][0]
            if token_input == "WETH":
                token_input = "ETH"
            if token_output == "WETH":
                token_output = "ETH"
            if token_input == "WBTC":
                token_input = "BTC"
            if token_output == "WBTC":
                token_output = "BTC"

            sample_prices = {}

            # find the direction
            if f"{token_input}{token_output}" in binance_prices:
                max_price, min_price = find_binance_price(user_request_time-12, user_request_time, binance_prices[f"{token_input}{token_output}"])

                for sample in SAMPLES:
                    gaussian_samples = generate_gaussian(max_price, min_price, num_samples=sample)
                    pareto_samples = generate_pareto(max_price, min_price, num_samples=sample)

                    sample_prices[sample] = (gaussian_samples, pareto_samples)

                cowswap_settled_price = settled_amount_out * 10**ADDR_TO_COINS[token_input_addr][1]/ settled_amount_in / 10**ADDR_TO_COINS[token_output_addr][1]
            elif f"{token_output}{token_input}" in binance_prices:
                delta_x = -token_output_amount / 10**ADDR_TO_COINS[token_output_addr][1]
                delta_y = token_input_amount / 10**ADDR_TO_COINS[token_input_addr][1]
                max_price, min_price = find_binance_price(user_request_time-12, user_request_time, binance_prices[f"{token_output}{token_input}"])

                for sample in SAMPLES:
                    gaussian_samples = generate_gaussian(-min_price, -max_price, num_samples=sample)
                    gaussian_samples = [-i for i in gaussian_samples]
                    pareto_samples = generate_pareto(-min_price, -max_price, num_samples=sample)
                    pareto_samples = [-i for i in pareto_samples]

                    sample_prices[sample] = (gaussian_samples, pareto_samples)

                cowswap_settled_price = settled_amount_in * 10**ADDR_TO_COINS[token_output_addr][1] / settled_amount_out / 10**ADDR_TO_COINS[token_input_addr][1]
            else:
                # print(f"Price not found for {token_input} and {token_output}")
                continue

            
            cost = row["transaction_fees_usd"]
            fee_detials = json.loads(row["fee_details"])
            if fee_detials["receipt_effective_gas_price"] - fee_detials["base_fee_per_gas"] > 10**9:
                cost = cost * (fee_detials["base_fee_per_gas"] + 10**9) / fee_detials["receipt_effective_gas_price"]
            
            cost = cost / row["count"]
    
            for fee in FEES:
                result = results[fee]
                token_input_amount_after_fee = round(token_input_amount * (1 - fee))

                if f"{token_input}{token_output}" in binance_prices:
                    delta_x = token_input_amount_after_fee / 10**ADDR_TO_COINS[token_input_addr][1]
                    delta_y = -token_output_amount / 10**ADDR_TO_COINS[token_output_addr][1]
                elif f"{token_output}{token_input}" in binance_prices:
                    delta_x = -token_output_amount / 10**ADDR_TO_COINS[token_output_addr][1]
                    delta_y = token_input_amount_after_fee / 10**ADDR_TO_COINS[token_input_addr][1]

                for sample in SAMPLES:
                    gaussian_samples, pareto_samples = sample_prices[sample]
                    gaussian_searcher_profit, gaussian_user_kickback = compute_delta(delta_x, delta_y, gaussian_samples)
                    pareto_searcher_profit, pareto_user_kickback = compute_delta(delta_x, delta_y, pareto_samples)

                    if gaussian_user_kickback < 0:
                        gaussian_user_kickback = 0
                    if pareto_user_kickback < 0:
                        pareto_user_kickback = 0

                    if token_output_addr in STABLE_COIN_ADDRESSES:
                        cowswap_amount_usd = settled_amount_out / 10**ADDR_TO_COINS[token_output_addr][1]
                        simulated_amount_usd = token_output_amount / 10**ADDR_TO_COINS[token_output_addr][1]
                    else:
                        cowswap_amount_usd = settled_amount_out * min_price / 10**ADDR_TO_COINS[token_output_addr][1]
                        simulated_amount_usd = token_output_amount  * min_price / 10**ADDR_TO_COINS[token_output_addr][1]


                    final_amount_usd_gaussian = simulated_amount_usd + gaussian_user_kickback
                    final_amount_usd_pareto = simulated_amount_usd + pareto_user_kickback

                    gaussian_ratio = 100*(final_amount_usd_gaussian-cowswap_amount_usd-cost)/cowswap_amount_usd
                    pareto_ratio = 100*(final_amount_usd_pareto-cowswap_amount_usd-cost)/cowswap_amount_usd
                     
                    result["gaussian"][sample].append((gaussian_ratio, final_amount_usd_gaussian, cowswap_amount_usd, cost, gaussian_samples[1], gaussian_searcher_profit))
                    result["pareto"][sample].append((pareto_ratio, final_amount_usd_pareto, cowswap_amount_usd, cost, pareto_samples[1], pareto_searcher_profit))
            prices.append([max_price, min_price, cowswap_settled_price, token_input, token_output, month_str, row["block_timestamp"], row["transaction_fees_usd"], row["transaction_hash"]])
    return prices, results


def main():
    print(f"🚀🚀🚀 Simulating... (CoWSwap)")
    t = time.time()
    prices, results = simulate()
    print(f"✅ Time taken: {time.time()-t:.2f}s")

    data = []
    for fee, result in results.items():
        for sample in SAMPLES:
            gaussian_counts = result["gaussian"][sample]
            new_gaussian_counts = [i for i in gaussian_counts ]
            positive_gaussian_counts = [i for i in new_gaussian_counts if i[0] > 0]
            pareto_counts = result["pareto"][sample]
            new_pareto_counts = [i for i in pareto_counts]
            positive_pareto_counts = [i for i in new_pareto_counts if i[0] > 0]

            data.append((100*fee, sample, 100*len(positive_gaussian_counts)/len(new_gaussian_counts), "Gaussian"))
            data.append((100*fee, sample, 100*len(positive_pareto_counts)/len(new_pareto_counts), "Pareto"))
    
    df = pd.DataFrame(data, columns=["Fee", "Arbitrageurs (#)", "Ratio", "Distribution"])
    print("🔥 Plotting...")
    output_figure_path = os.path.join(FIGURES_PATH, "cowswap-ratio.pdf")
    plot(df, output_figure_path)
    print(f"🎉 Done! Figure is in {output_figure_path}")


if __name__ == "__main__":
    main()
