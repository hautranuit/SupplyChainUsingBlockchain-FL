# IPFS Connector for ChainFLIP FL Integration
# This module will handle connections to an IPFS gateway
# and provide functions to retrieve data from IPFS given a CID.

import requests
import time

# Configuration
IPFS_GATEWAY_URL = "https://ipfs.io/ipfs/"  # Using a public gateway, can be replaced with a project-specific one
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

class IPFSConnector:
    def __init__(self, gateway_url=IPFS_GATEWAY_URL):
        self.gateway_url = gateway_url
        if not self.gateway_url.endswith("/"):
            self.gateway_url += "/"

    def fetch_data_from_cid(self, cid):
        """Fetches data from IPFS given a CID."""
        if not cid or not isinstance(cid, str):
            print("Error: Invalid CID provided.")
            return None

        url = f"{self.gateway_url}{cid}"
        print(f"Attempting to fetch data from IPFS: {url}")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                
                # Assuming the content is text-based (e.g., JSON, plain text)
                # For binary data, response.content would be used
                # Consider adding content-type checks if various data types are expected
                print(f"Successfully fetched data for CID {cid}. Status: {response.status_code}")
                return response.text # Or response.json() if content is always JSON, or response.content for binary
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error fetching CID {cid} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                # Specific handling for 404 Not Found, as it might not be a transient error
                if e.response.status_code == 404:
                    print(f"CID {cid} not found on IPFS gateway. No further retries.")
                    return None 
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error fetching CID {cid} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            except requests.exceptions.Timeout as e:
                print(f"Timeout fetching CID {cid} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            except requests.exceptions.RequestException as e:
                print(f"An unexpected error occurred fetching CID {cid} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to fetch CID {cid} after {MAX_RETRIES} attempts.")
                return None
        return None # Should be unreachable if loop logic is correct

# Example Usage (for testing this module independently)
if __name__ == '__main__':
    print("Testing IPFSConnector...")
    connector = IPFSConnector()

    # Example CIDs (replace with actual CIDs from your project for testing)
    # A well-known public CID for testing (e.g., the IPFS logo or a sample text file)
    # test_cid_valid = "QmbWqxBEKC3P8tqsKc98xmWNzrzDtRLMiMPL8wBuTGsMnR" # Example: IPFS whitepaper
    test_cid_valid_text = "QmPZ9gcCEpqKTo6aq61g2nXGUhM4iCL3ewB6LDXZCtioEB" # A simple text file CID: "Hello IPFS!"
    test_cid_invalid_format = "not_a_cid"
    test_cid_non_existent = "Qmaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    print(f"\n--- Test Case 1: Valid Text CID ---")
    data = connector.fetch_data_from_cid(test_cid_valid_text)
    if data:
        print(f"Data fetched for {test_cid_valid_text}:\n{data[:200]}...") # Print first 200 chars
    else:
        print(f"Failed to fetch data for {test_cid_valid_text}.")

    print(f"\n--- Test Case 2: Invalid CID Format ---")
    data_invalid = connector.fetch_data_from_cid(test_cid_invalid_format)
    if data_invalid is None:
        print(f"Correctly handled invalid CID format: {test_cid_invalid_format}")
    else:
        print(f"Incorrectly processed invalid CID: {test_cid_invalid_format}")

    print(f"\n--- Test Case 3: Non-Existent CID ---")
    data_non_existent = connector.fetch_data_from_cid(test_cid_non_existent)
    if data_non_existent is None:
        print(f"Correctly handled non-existent CID: {test_cid_non_existent}")
    else:
        print(f"Fetched unexpected data for non-existent CID: {test_cid_non_existent}")

    # Test with a different gateway if needed
    # print("\n--- Test Case 4: Custom Gateway (example, not functional without a real gateway) ---")
    # custom_connector = IPFSConnector(gateway_url="http://localhost:8080/ipfs/")
    # data_custom = custom_connector.fetch_data_from_cid(test_cid_valid_text)
    # if data_custom:
    #     print(f"Data fetched via custom gateway: {data_custom}")
    # else:
    #     print(f"Failed to fetch via custom gateway.")

    print("\nIPFSConnector testing finished.")

