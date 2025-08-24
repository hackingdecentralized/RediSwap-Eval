import json
import os
import pandas as pd
import plotly.graph_objects as go
import time
import zipfile

from plotly.subplots import make_subplots

from simulation.base import *
from utils import *


# Destination path
FIGURES_PATH = os.path.join(PARENT_DIR, "figures")
UNISWAPX_PATH = os.path.join(DATA_PATH, "uniswapx")

  
# Constants
FEES = [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
SAMPLES = [3, 10, 20]


def load_uniswapx(month_str):
    month_json_path = os.path.join(UNISWAPX_PATH, f"{month_str}.json.zip")
    if not os.path.exists(month_json_path):
        return None
    
    with zipfile.ZipFile(month_json_path, "r") as zipf:
        json_filename = f"{month_str}.json"
        with zipf.open(json_filename) as f:
            order_hashes = json.load(f)
    
    # Load the corresponding CSV file
    month_csv_path = os.path.join(UNISWAPX_PATH, f"{month_str}.csv.zip")
    df = pd.read_csv(month_csv_path, compression="zip")

    return df, order_hashes


def simulate():
    date_ranges = pd.date_range(start="2023-09-01", end="2024-08-31", freq="MS")
    results = {}
    prices = []

    for fee in FEES:
        results[fee] = {
            "guassian": {k: [] for k in SAMPLES},
            "pareto": {k: [] for k in SAMPLES},
        }

    for month in date_ranges:
        month_str = month.strftime("%Y%m")
        print(f"Processing {month_str}")
        binance_prices = load_monthly_prices(month_str)
        uniswapx_data_df, uniswapx_data_map = load_uniswapx(month_str)
        uniswapx_data_df["block_timestamp"] = pd.to_datetime(uniswapx_data_df["block_timestamp"])

        for _, row in uniswapx_data_df.iterrows():
            order_hash = row["order_hash"]
            item = uniswapx_data_map[order_hash]
            user_request_time = int(item["createdAt"])

            order_input = item["input"]
            order_outputs = item["outputs"]
            swapper = item["swapper"]
            
            # Check if the order output is the swapper
            order_output = order_outputs[0]
            if order_output["recipient"] != swapper:
                print(f"Order output not found for {order_hash}")
                continue

            token_input_best_amount = int(order_input["startAmount"])
            token_input_amount = int(order_input["endAmount"])
            token_output_amount = int(order_output["endAmount"])
            token_output_best_amount = int(order_output["startAmount"])

            if "settledAmounts" not in item:
                continue
            
            settled_amount = item["settledAmounts"][0]
            token_input_addr = settled_amount["tokenIn"]
            token_output_addr = settled_amount["tokenOut"]
            
            if token_input_addr not in ADDR_TO_COINS or token_output_addr not in ADDR_TO_COINS:
                continue

            # waiting time
            current_block_timestamp = find_block_timestamp(user_request_time)
            wait_block = (row["block_timestamp"] - current_block_timestamp).total_seconds()/12
            # transaction fee
            cost = row["transaction_fees_usd"]
            fee_detials = json.loads(row["fee_details"])
            # only pay 1Gwei more than the base fee
            if fee_detials["receipt_effective_gas_price"] - fee_detials["base_fee_per_gas"] > 10**9:
                cost = cost * (fee_detials["base_fee_per_gas"] + 10**9) / fee_detials["receipt_effective_gas_price"]

            settled_amount_out = int(settled_amount["amountOut"])
            settled_amount_in = int(settled_amount["amountIn"])

            if len(order_outputs) > 1:
                if order_outputs[1]["recipient"] == swapper and order_outputs[1]["token"] == order_output["token"]:
                    token_output_amount += int(order_outputs[1]["endAmount"])
                    token_output_best_amount += int(order_outputs[1]["startAmount"])
                    settled_amount_out += int(item["settledAmounts"][1]["amountOut"])
                    settled_amount_in += int(item["settledAmounts"][1]["amountIn"])

            # Token symbols
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
            if f"{token_input}{token_output}" in binance_prices:
                max_price, min_price = find_binance_price(user_request_time, current_block_timestamp.timestamp()+12, binance_prices[f"{token_input}{token_output}"])

                for sample in SAMPLES:
                    gaussian_samples = generate_gaussian(max_price, min_price, num_samples=sample)
                    pareto_samples = generate_pareto(max_price, min_price, num_samples=sample)

                    sample_prices[sample] = (gaussian_samples, pareto_samples)

                uniswapx_settled_price = settled_amount_out * 10**ADDR_TO_COINS[token_input_addr][1]/ settled_amount_in / 10**ADDR_TO_COINS[token_output_addr][1]
                uniswapx_worst_price = token_output_amount * 10**ADDR_TO_COINS[token_input_addr][1] / token_input_amount / 10**ADDR_TO_COINS[token_output_addr][1]
                uniswapx_best_price = token_output_best_amount * 10**ADDR_TO_COINS[token_input_addr][1] / token_input_best_amount / 10**ADDR_TO_COINS[token_output_addr][1]
            elif f"{token_output}{token_input}" in binance_prices:
                max_price, min_price = find_binance_price(user_request_time, current_block_timestamp.timestamp()+12, binance_prices[f"{token_output}{token_input}"])

                for sample in SAMPLES:
                    gaussian_samples = generate_gaussian(-min_price, -max_price, num_samples=sample)
                    gaussian_samples = [-i for i in gaussian_samples]
                    pareto_samples = generate_pareto(-min_price, -max_price, num_samples=sample)
                    pareto_samples = [-i for i in pareto_samples]

                    sample_prices[sample] = (gaussian_samples, pareto_samples)

                uniswapx_settled_price = settled_amount_in * 10**ADDR_TO_COINS[token_output_addr][1] / settled_amount_out / 10**ADDR_TO_COINS[token_input_addr][1]
                uniswapx_worst_price = token_input_amount * 10**ADDR_TO_COINS[token_output_addr][1] / token_output_amount / 10**ADDR_TO_COINS[token_input_addr][1]
                uniswapx_best_price = token_input_best_amount * 10**ADDR_TO_COINS[token_output_addr][1] / token_output_best_amount / 10**ADDR_TO_COINS[token_input_addr][1]
            else:
                # print(f"Price not found for {token_0}-{token_1} or {token_1}-{token_0}")
                continue

            if token_input_best_amount == token_input_amount:
                if token_output_best_amount == token_output_amount:
                    uniswapx_ratio = 1
                else:
                    uniswapx_ratio = (settled_amount_out - token_output_amount) / (token_output_best_amount - token_output_amount)
            else:
                uniswapx_ratio = 1 - (settled_amount_in - token_input_best_amount) / (token_input_amount - token_input_best_amount)

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
                    guassian_samples, pareto_samples = sample_prices[sample]
                    guassian_searcher_profit, guassian_user_kickback = compute_delta(delta_x, delta_y, guassian_samples)
                    pareto_searcher_profit, pareto_user_kickback = compute_delta(delta_x, delta_y, pareto_samples)

                    if guassian_user_kickback < 0:
                        guassian_user_kickback = 0
                    if pareto_user_kickback < 0:
                        pareto_user_kickback = 0

                    if settled_amount["tokenOut"] in STABLE_COIN_ADDRESSES:
                        uniswapx_amount_usd = settled_amount_out / 10**ADDR_TO_COINS[settled_amount["tokenOut"]][1]
                        simulated_amount_usd = token_output_amount / 10**ADDR_TO_COINS[settled_amount["tokenOut"]][1]
                    else:
                        uniswapx_amount_usd = settled_amount_out * min_price / 10**ADDR_TO_COINS[settled_amount["tokenOut"]][1]
                        simulated_amount_usd = token_output_amount  * min_price / 10**ADDR_TO_COINS[settled_amount["tokenOut"]][1]

                    final_amount_usd_guassian = simulated_amount_usd + guassian_user_kickback
                    final_amount_usd_pareto = simulated_amount_usd + pareto_user_kickback

                    guassian_ratio = 100*(final_amount_usd_guassian-uniswapx_amount_usd-cost)/uniswapx_amount_usd
                    pareto_ratio = 100*(final_amount_usd_pareto-uniswapx_amount_usd-cost)/uniswapx_amount_usd
                     
                    result["guassian"][sample].append((guassian_ratio, final_amount_usd_guassian, uniswapx_amount_usd, cost, guassian_samples[1], guassian_searcher_profit))
                    result["pareto"][sample].append((pareto_ratio, final_amount_usd_pareto, uniswapx_amount_usd, cost, pareto_samples[1], pareto_searcher_profit))

            prices.append([max_price, min_price, uniswapx_best_price, uniswapx_worst_price, uniswapx_settled_price, uniswapx_ratio, token_input, token_output, month_str, row["block_timestamp"], fee_detials["receipt_effective_gas_price"], fee_detials["base_fee_per_gas"], order_hash, wait_block, month_str])
    return prices, results


# ("Binance Price", "UniswapX Settled Price", "UniswapX Best Price", "UniswapX Worst Price")
def plot_kline(titles, candlestick_dfs, y_range, y_title, filename):
    # Create a 2x2 subplot figure
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, 
                        subplot_titles=titles,
                        vertical_spacing=0.11, horizontal_spacing=0.11)

    for binance_candlestick_df, indexes in zip(candlestick_dfs, [(1, 1), (1, 2), (2, 1), (2, 2)]):
        fig.add_trace(go.Candlestick(x=binance_candlestick_df.index,
                                    open=binance_candlestick_df['Open'],
                                    high=binance_candlestick_df['High'],
                                    low=binance_candlestick_df['Low'],
                                    close=binance_candlestick_df['Close'],
                                    increasing_line_color='green', decreasing_line_color='red'),
                    row=indexes[0], col=indexes[1])
        
    # Update the layout for title, axis labels, plot size, and unified y-axis range
    fig.update_layout(title=None,
                        xaxis_title=None,
                    yaxis_title=y_title,
                    xaxis_showticklabels=True, 
                    xaxis2_showticklabels=True,
                    height=1000, width=1600, showlegend=False,
                    xaxis_rangeslider_visible=False,  # Remove the range slider
                    yaxis_range=y_range,  # Set y-axis range for all charts
                    yaxis2_title=None,
                    xaxis2_rangeslider_visible=False,
                    yaxis2_range=y_range,  # Set y-axis range for second subplot
                    yaxis3_title=y_title,
                    xaxis3_rangeslider_visible=False,
                    yaxis3_range=y_range,  # Set y-axis range for third subplot
                    yaxis4_title=None,
                    xaxis4_rangeslider_visible=False,
                    yaxis4_range=y_range)  # Set y-axis range for fourth subplot

    fig.update_layout(
        title=None,
        xaxis_title=None,
        annotations=[
            dict(font=dict(size=36)) for annotation in fig['layout']['annotations']
        ],
        height=1000, width=1600, showlegend=False,
        xaxis_rangeslider_visible=False,  # Remove the range slider
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        xaxis4_rangeslider_visible=False,
        xaxis_showticklabels=True, 
        xaxis2_showticklabels=True,
        yaxis_range=y_range,
        yaxis2_range=y_range,
        yaxis3_range=y_range,
        yaxis4_range=y_range,

        # Y-axis titles with larger font sizes
        yaxis_title=dict(text=y_title, font=dict(size=32)),
        yaxis3_title=dict(text=y_title, font=dict(size=32)),

        # Larger tick fonts
        xaxis_tickfont=dict(size=32),
        yaxis_tickfont=dict(size=32),
        xaxis2_tickfont=dict(size=32),
        yaxis2_tickfont=dict(size=32),
        xaxis3_tickfont=dict(size=32),
        yaxis3_tickfont=dict(size=32),
        xaxis4_tickfont=dict(size=32),
        yaxis4_tickfont=dict(size=32),

        xaxis=dict(tickformat="%y-%m"),     # Set custom date format as YY-MM
        xaxis2=dict(tickformat="%y-%m"),    # For second subplot
        xaxis3=dict(tickformat="%y-%m"),    # For third subplot
        xaxis4=dict(tickformat="%y-%m")     # For fourth subplot
    )

    fig.write_image(filename)


def main():
    print(f"🚀🚀🚀 Simulating... (UniswapX)")
    t = time.time()
    prices, results = simulate()
    print(f"✅ Time taken: {time.time()-t:.2f}s")

    data = []
    for fee, result in results.items():
        for sample in SAMPLES:
            guassian_counts = result["guassian"][sample]
            positive_guassian_counts = [i for i in guassian_counts if i[0] > 0]
            pareto_counts = result["pareto"][sample]
            positive_pareto_counts = [i for i in pareto_counts if i[0] > 0]

            data.append((100*fee, sample, 100*len(positive_guassian_counts)/len(guassian_counts), "Gaussian"))
            data.append((100*fee, sample, 100*len(positive_pareto_counts)/len(pareto_counts), "Pareto"))
    df = pd.DataFrame(data, columns=["Fee", "Arbitrageurs (#)", "Ratio", "Distribution"])
    print("🔥 Plotting...")
    output_figure_path = os.path.join(FIGURES_PATH, "uniswapx-ratio.pdf")
    plot(df, output_figure_path)
    print(f"🎉 Done! Figure is in {output_figure_path}")

    print("🔥 Plotting Kline Chart...")
    prices_df = pd.DataFrame(prices, columns=["max_price", "min_price", "uniswapx_best_price", "uniswapx_worst_price", "uniswapx_settled_price", "uniswapx_ratio", "token_input", "token_output", "month_str", "block_timestamp", "receipt_effective_gas_price", "base_fee_per_gas", "order_hash", "wait_time", "month_str"])
    # USDT-ETH
    usdt_eth_prices_df = prices_df[(prices_df["token_input"]=="USDT")&(prices_df["token_output"]=="ETH")].reindex()
    usdt_y_title = "Price of ETH (USDT)"
    usdt_y_range = [1500, 7000]
    usdt_output_figure_path = os.path.join(FIGURES_PATH, "usdt-eth.pdf")
    # ETH-USDT
    eth_usdt_prices_df = prices_df[(prices_df["token_input"]=="ETH")&(prices_df["token_output"]=="USDT")].reindex()
    eth_y_range = [1000, 4300]
    eth_y_title = "Price of ETH (USDT)"
    usdd_output_figure_path = os.path.join(FIGURES_PATH, "eth-usdt.pdf")

    # Plot the candlestick chart
    for token_prices_df, y_range, y_title, output_figure_path in [(usdt_eth_prices_df, usdt_y_range, usdt_y_title, usdt_output_figure_path), (eth_usdt_prices_df, eth_y_range, eth_y_title, usdd_output_figure_path)]:
        binance_grouped = token_prices_df.groupby(token_prices_df['block_timestamp'].dt.date).agg({
            'max_price': ['first', 'max', 'last'],  # 'first' as Open, 'last' as Close, 'max' as High, 'min' as Low
            'min_price': ['min'],
        })
        binance_grouped.columns = ["Open", "High", "Close", "Low"]

        other_grouped = [
            token_prices_df.groupby(token_prices_df['block_timestamp'].dt.date).agg({
                i : ['first', 'max', 'min', 'last'],  # 'first' as Open, 'last' as Close, 'max' as High, 'min' as Low
            })
            for i in ['uniswapx_settled_price', 'uniswapx_best_price', 'uniswapx_worst_price']
        ]
        for grouped in other_grouped:
            grouped.columns = ["Open", "High", "Low", "Close"]

        candlestick_dfs = []
        for grouped in [binance_grouped] + other_grouped:
            candlestick_df = pd.DataFrame({
                'Date': grouped.index,
                'Open': grouped['Open'],  # First price of the day
                'High': grouped['High'],  # Max price of the day
                'Low': grouped['Low'],    # Min price of the day
                'Close': grouped['Close'] # Last price of the day
            })
            candlestick_dfs.append(candlestick_df)

        subplot_titles=("Binance Price", "UniswapX Settled Price", "UniswapX Best Price", "UniswapX Worst Price")
        plot_kline(subplot_titles, candlestick_dfs, y_range, y_title, output_figure_path)
        print(f"🎉 Done! Figure is in {output_figure_path}")


if __name__ == "__main__":
    main()
