import sys
import argparse
import bittensor as bt
import os
from dotenv import load_dotenv
import time

def main():
    load_dotenv()
    
    burn_rate = 0.8
    
    # 1) Parse the single argument for target_uid
    parser = argparse.ArgumentParser(description="Set weights to target UID and burn burn_rate to UID 0.")
    parser.add_argument('--target_uid', type=int, required=True,
                        help="Target UID to receive weight after burn rate goes to UID 0.")
    parser.add_argument('--wallet_name', type=str, required=True,
                        help="The name of the wallet to use.")
    parser.add_argument('--wallet_hotkey', type=str, required=True,
                        help="The hotkey to use for the wallet.")

    args = parser.parse_args()

    NETUID = 68
    
    wallet = bt.wallet(
        name=args.wallet_name,  
        hotkey=args.wallet_hotkey, 
    )

    # Create Subtensor connection using network from .env
    subtensor_network = os.getenv('SUBTENSOR_NETWORK')
    subtensor = bt.subtensor(network=subtensor_network)


    # Download the metagraph for netuid=68
    metagraph = subtensor.metagraph(NETUID)

    # Check registration
    hotkey_ss58 = wallet.hotkey.ss58_address
    if hotkey_ss58 not in metagraph.hotkeys:
        print(f"Hotkey {hotkey_ss58} is not registered on netuid {NETUID}. Exiting.")
        sys.exit(1)

    # 2) Build the weight vector
    n = len(metagraph.uids)
    weights = [0.0] * n

    # Validate the user-provided target UID
    if not (0 <= args.target_uid < n):
        print(f"Error: target_uid {args.target_uid} out of range [0, {n-1}]. Exiting.")
        sys.exit(1)

    # Set weights: burn to UID 0, remainder to target
    weights[0] = burn_rate
    weights[args.target_uid] += 1.0 - burn_rate

    # 3) Send the weights to the chain with retry logic
    max_retries = 10
    delay_between_retries = 12  # seconds
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to set weights.")
            result = subtensor.set_weights(
                netuid=NETUID,
                wallet=wallet,
                uids=metagraph.uids,
                weights=weights,
                wait_for_inclusion=True
            )
            print(f"Result from set_weights: {result}")

            # Only break if result indicates success (result[0] == True).
            if result[0] is True:
                print("Weights set successfully. Exiting retry loop.")
                break
            else:
                print("set_weights returned a non-success response. Will retry if attempts remain.")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay_between_retries} seconds...")
                    time.sleep(delay_between_retries)

        except Exception as e:
            print(f"Error setting weights: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                print("Failed to set weights after multiple attempts. Exiting.")
                sys.exit(1)

    print("Done.")

if __name__ == "__main__":
    main()
