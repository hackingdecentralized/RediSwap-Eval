# RediSwap

This repository includes the code and data for reproducibility of the paper: "RediSwap: MEV Redistribution Mechanism for CFMMs". 

## Setup

To set up the environment, ensure you have `Python` installed. You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Data

### Binance Price Data

To download price data from Binance, run the script `get_binance_data.py`. It will automatically fetch and save the data in the `data/binance` folder.

### CoWSwap

The `cowswap` folder contains trades executed through CoWSwap from September 2023 to August 2024, compressed into separate archives for each month. As mentioned in the paper, each file only includes the trades involving tokens from the list: `[USDC, USDT, DAI, WBTC, BTC, ETH, WETH, PEPE, MATIC, LINK]`.

The file `cowswap_count.csv` provides a monthly count of trades executed through CoWSwap that utilized the liquidity pool.

### UniswapX

The `uniswapx` folder contains trades executed through UniswapX from September 2023 to August 2024, compressed into separate archives for each month. As mentioned in the paper, each file only includes the trades involving tokens from the list: `[USDC, USDT, DAI, WBTC, BTC, ETH, WETH, PEPE, MATIC, LINK]`.

The file `uniswapx_count.csv` provides a monthly count of trades executed through UniswapX that utilized the liquidity pool.

### Uniswap V2

The files `UniswapV2-USDC.csv.zip` and `UniswapV2-USDT.csv.zip` include the liquidity data for the USDC-ETH and USDT-ETH pools on Uniswap V2 (Ethereum). The data spans the period from September 2023 to August 2024, with liquidity recorded at each block.


## Code

To generate the figures shown in the paper, you can navigate to each corresponding subdirectory and execute the scripts.

### Simulation

There are three scripts in the `simulation` folder that generate the figures in Section 5:

- `compare_cowswap.py`: Fig. 2 (a). Percentage of orders with execution prices better than those on CoWSwap. After running the script, it will generate the following figure in the `figures` folder.
    - `cowswap-ratio.pdf`
- `compare_uniswapx.py`:
    - Fig. 2 (b). Percentage of orders with execution prices better than those on UniswapX.
    - Fig. 3. Candlestick charts of Binance and UniswapX ETH/USDT prices over time.
    
    After running the script, it will generate the following figures in the `figures` folder.
    - `eth-usdt.pdf`
    - `usdt-eth.pdf`
- `lvr.py`: Fig. 4. The CDF of LVR reduction for WETH-USDC and WETH-USDT using RediSwap. After running the script, it will generate the following figures in the `figures` folder.
    - `eth-usdc-cdf.pdf`
    - `eth-usdt-cdf.pdf`


Besides, `base.py` and `utils.py` contain helper functions and constants used by the scripts mentioned above. 

### Appendix

There are two scripts in the `appendix` folder that generate the figures in Appendix A:
- `searcher_without_pools.py`: Fig. 5 - Percentage of orders filled by direct exchange between users and solvers. After running the script, it will generate the following figures in the `figures` folder.
    - `figures/cowswap_searcher_without_pools.pdf`
    - `figures/uniswapx_searcher_without_pools.pdf`
- `trade_per_tx.py`: Fig. 6 - Distribution of the number of orders per batch. After running the script, it will generate the following figures in the `figures` folder.
    - `figures/cowswap_order_per_tx.pdf`
    - `figures/uniswapx_order_per_tx.pdf`
