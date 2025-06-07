import os
import glob
import json
import re
import logging
from datetime import datetime
from typing import  List, Any, Optional # Replaced with modern builtins
from typing import Any, Optional # Keep Any and Optional
import threading
from word2number import w2n
from collections import Counter


# LangChain Core

from langchain.prompts import  ChatPromptTemplate
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document # Removed - RAG Not Used
from langchain.text_splitter import RecursiveCharacterTextSplitter# Removed - RAG Not Used
# LangChain Community (models & utilities)
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader # Removed - RAG Not Used

#
# Chroma (Vector Store)
from langchain.vectorstores import Chroma

# UI
import gradio as gr

## imports
from src.common.common import Embeddings,LLM_INSTANCE
from src.common.constant import BASE_PATH,Data_PATH,CHROMA_DB_PATH

# Create 'logs' directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/server_monitoring.log", mode='a'),  # Log file
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# Test the logging setup


# Global instances
llm_instance = None
vectorstore = None # Removed - Unused

# (Keep all your existing data processing and tool functions as they are)
# ... (previous code from process_server_data down to RAG functions) ...
# -----------------------------
# Load JSON at startup
# -----------------------------
try:
    with open(Data_PATH, "r") as file:
        server_data_raw = json.load(file)
        logger.info(f"Loaded data for {len(server_data_raw)} servers")
except FileNotFoundError:
    logger.error("Server data file 'new_dump1.json' not found. Please ensure it's in the same directory.")
    server_data_raw = []
except json.JSONDecodeError:
    logger.error("Invalid JSON format in server data file 'new_dump1.json'.")
    raise Exception("Invalid JSON format in server data file.")


# -----------------------------
# CO2 and Energy Efficiency Constants
# -----------------------------
DEFAULT_CARBON_INTENSITY = {
    'low_carbon_grid': 0.1,
    'average_grid': 0.5,
    'high_carbon_grid': 0.8
}

EFFICIENCY_THRESHOLDS = {
    'cpu_power_ratio': {
        'excellent': 0.3,
        'good': 0.5,
        'average': 0.7,
        'poor': 1.0
    },
    'thermal_management': {
        'optimal_temp_range': (15, 25)
    }
}

def estimate_power(cpu_util: float) -> float:
    base_power = 50
    max_power = 200
    return base_power + ((max_power - base_power) * cpu_util / 100.0)

# -----------------------------
# Data Processing Functions
# -----------------------------
def process_server_data() -> dict[str, dict]: # Replaced typing.Dict with dict
    processed_data = {}
    if not server_data_raw:
        logger.warning("No raw server data to process.")
        return processed_data

    logger.info(f"Processing data for {len(server_data_raw)} servers")

    for server_item in server_data_raw:
        serial_number = server_item.get("serial_number")
        if not serial_number:
            continue

        power_data_entries = server_item.get("power", [])
        if not power_data_entries:
            continue

        cpu_utils = []
        temps = []
        peaks = []
        timestamps = []
        records = []
        estimated_energy_kwh = 0.0
        estimated_powers = []

        valid_entries_for_sorting = []
        for entry in power_data_entries:
            timestamp_str = entry.get("time", "")
            if timestamp_str:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M:%S")
                    valid_entries_for_sorting.append((timestamp, entry))
                except ValueError:
                    logger.warning(f"Invalid timestamp format for server {serial_number}, entry skipped: {timestamp_str}")
                    continue

        valid_entries_for_sorting.sort(key=lambda x: x[0])
        sorted_power_data_entries = [item[1] for item in valid_entries_for_sorting]


        for idx, entry in enumerate(sorted_power_data_entries):
            try:
                timestamp_str = entry.get("time", "")
                timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M:%S")

                cpu_util = entry.get("cpu_util")
                amb_temp = entry.get("amb_temp")
                peak = entry.get("peak")

                est_power_val = None
                if isinstance(cpu_util, (int, float)):
                    est_power_val = estimate_power(float(cpu_util))
                    estimated_powers.append(est_power_val)

                if idx > 0 and est_power_val is not None and timestamps: # Check est_power_val is not None
                    prev_time = timestamps[-1]
                    delta_hours = (timestamp - prev_time).total_seconds() / 3600.0
                    if delta_hours > 0:
                         estimated_energy_kwh += (est_power_val * delta_hours) / 1000.0 # est_power_val is in Watts, so this is Wh. Division by 1000 makes it kWh.

                if isinstance(cpu_util, (int, float)):
                    cpu_utils.append(float(cpu_util))
                if isinstance(amb_temp, (int, float)):
                    temps.append(float(amb_temp))
                if isinstance(peak, (int, float)):
                    peaks.append(float(peak))

                timestamps.append(timestamp)
                records.append({
                    "time": timestamp,
                    "time_str": timestamp_str,
                    "cpu_util": cpu_util,
                    "amb_temp": amb_temp,
                    "peak": peak,
                    "power_consumption": entry.get("power_consumption"),
                    "temperature": entry.get("temperature"),
                    "fan_speed": entry.get("fan_speed"),
                    "cpu_watts": entry.get("cpu_watts"),
                    "dimm_watts": entry.get("dimm_watts"),
                    "estimated_power": est_power_val
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing entry for server {serial_number}: {e}")
                continue

        if not records:
            continue

        avg_cpu = round(sum(cpu_utils) / len(cpu_utils), 2) if cpu_utils else None
        peak_cpu_util_val = max(cpu_utils) if cpu_utils else None
        peak_cpu_record = next((r for r in records if r["cpu_util"] == peak_cpu_util_val), None) if peak_cpu_util_val is not None else None
        lowest_cpu_util_val = min(cpu_utils) if cpu_utils else None
        lowest_cpu_record = next((r for r in records if r["cpu_util"] == lowest_cpu_util_val), None) if lowest_cpu_util_val is not None else None

        avg_est_power = round(sum(estimated_powers) / len(estimated_powers), 2) if estimated_powers else None


        max_amb_temp = max(temps) if temps else None
        max_temp_record = next((r for r in records if r["amb_temp"] == max_amb_temp), None) if max_amb_temp is not None else None
        min_amb_temp = min(temps) if temps else None
        min_temp_record = next((r for r in records if r["amb_temp"] == min_amb_temp), None) if min_amb_temp is not None else None

        max_peak = max(peaks) if peaks else None
        max_peak_record = next((r for r in records if r["peak"] == max_peak), None) if max_peak is not None else None
        min_peak = min(peaks) if peaks else None
        min_peak_record = next((r for r in records if r["peak"] == min_peak), None) if min_peak is not None else None

        latest_record = max(records, key=lambda x: x["time"]) if records else None

        co2_emissions = {
            grid_type: round(estimated_energy_kwh * intensity, 3)
            for grid_type, intensity in DEFAULT_CARBON_INTENSITY.items()
        }

        processed_data[serial_number] = {
            "avg_cpu_util": avg_cpu,
            "peak_cpu_util": peak_cpu_util_val,
            "peak_cpu_record": peak_cpu_record,
            "lowest_cpu_util": lowest_cpu_util_val,
            "lowest_cpu_record": lowest_cpu_record,
            "avg_est_power": avg_est_power,
            "max_amb_temp": max_amb_temp,
            "max_temp_record": max_temp_record,
            "min_amb_temp": min_amb_temp,
            "min_temp_record": min_temp_record,
            "max_peak": max_peak,
            "max_peak_record": max_peak_record,
            "min_peak": min_peak,
            "min_peak_record": min_peak_record,
            "latest_record": latest_record,
            "estimated_energy_kwh": round(estimated_energy_kwh, 3),
            "co2_emissions": co2_emissions,
            "all_records": records
        }

    logger.info(f"Successfully processed data for {len(processed_data)} servers")
    return processed_data

processed_server_data = process_server_data()

server_rankings = {
    "top_cpu": sorted(
        [(k, v["peak_cpu_util"]) for k, v in processed_server_data.items() if v.get("peak_cpu_util") is not None],
        key=lambda x: x[1],
        reverse=True
    ),
    "bottom_cpu": sorted(
        [(k, v["lowest_cpu_util"]) for k, v in processed_server_data.items() if v.get("lowest_cpu_util") is not None],
        key=lambda x: x[1]
    ),
    "top_amb_temp": sorted(
        [(k, v["max_amb_temp"]) for k, v in processed_server_data.items() if v.get("max_amb_temp") is not None],
        key=lambda x: x[1],
        reverse=True
    ),
    "bottom_amb_temp": sorted(
        [(k, v["min_amb_temp"]) for k, v in processed_server_data.items() if v.get("min_amb_temp") is not None],
        key=lambda x: x[1]
    ),
    "top_peak": sorted(
        [(k, v["max_peak"]) for k, v in processed_server_data.items() if v.get("max_peak") is not None],
        key=lambda x: x[1],
        reverse=True
    ),
    "bottom_peak": sorted(
        [(k, v["min_peak"]) for k, v in processed_server_data.items() if v.get("min_peak") is not None],
        key=lambda x: x[1]
    )
}

logger.info("Server rankings initialized.")


# -----------------------------
# Tool Functions
# -----------------------------
def list_servers(_:str) -> str:
    if not processed_server_data:
        return "Error: No server data available."
    serials = list(processed_server_data.keys())
    if not serials:
        return "No servers found in the processed data."
    result = f"Available Servers ({len(serials)} total):\n" + "=" * 40 + "\n\n"
    for i, serial in enumerate(serials, 1):
        server_info = processed_server_data.get(serial)
        result += f"{i:2d}. Server {serial}:\n"
        if not server_info:
            result += "   - Data incomplete.\n\n"
            continue
        latest_rec = server_info.get("latest_record")
        time_str = latest_rec['time_str'] if latest_rec else "N/A"
        avg_cpu = server_info.get("avg_cpu_util")
        all_temps = [rec['amb_temp'] for rec in server_info.get("all_records", []) if rec.get('amb_temp') is not None]
        avg_temp = round(sum(all_temps) / len(all_temps), 1) if all_temps else None
        record_count = len(server_info.get("all_records", []))
        
        result += f"   - Last Seen: {time_str}\n"
        result += f"   - Total Records: {record_count}\n"
        
        avg_cpu_str = f"{avg_cpu:.1f}%" if avg_cpu is not None else "N/A"
        result += f"   - Average CPU: {avg_cpu_str}\n"
        
        avg_temp_str = f"{avg_temp:.1f}°C" if avg_temp is not None else "N/A"
        result += f"   - Average Ambient Temp: {avg_temp_str}\n"
        
        result += "\n" # Single blank line separator between servers
    return result


def extract_server_count(text: str, default: int = 10) -> float:
    if not text or not isinstance(text, str):
        return float(default)
    text_clean = text.lower().strip()
    if any(word in text_clean for word in ['all', 'every', 'everything']):
        return float('inf')
    
    digits = re.findall(r'\d+', text.replace(',', ''))
    if digits:
        try:
            return float(digits[0]) # Use the first number found
        except ValueError: 
            pass 

    try:
        # Keep this word removal conservative to avoid removing parts of numbers
        cleaned_words = [word for word in text_clean.split() if word not in {'top', 'show', 'give', 'me', 'the', 'first', 'last', 'highest', 'lowest', 'servers', 'server'}]
        if cleaned_words:
            try:
                return float(w2n.word_to_num(" ".join(cleaned_words)))
            except ValueError:
                for word in cleaned_words: 
                    try:
                        return float(w2n.word_to_num(word))
                    except ValueError:
                        continue
    except Exception as e: 
        logger.debug(f"Word to number conversion failed for '{text}': {e}")
    return float(default)

def get_top_servers_by_cpu_util(query: str = "") -> str:
    if not server_rankings.get("top_cpu"):
        return "No CPU utilization data available for ranking top servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["top_cpu"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for top CPU utilization."
    result_header = "Server with highest peak CPU utilization:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by highest peak CPU utilization:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by highest peak CPU utilization:\n\n"
    result = result_header
    for i, (serial, peak_cpu) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("peak_cpu_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        peak_record = server_info["peak_cpu_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Peak CPU: {peak_cpu}%\n"
        result += f"   - Timestamp: {peak_record['time_str']}\n"
        result += f"   - Power: {peak_record.get('power_consumption', 'N/A')}W\n"
        result += f"   - Temperature: {peak_record.get('temperature', 'N/A')}°C\n"
        result += f"   - Fan Speed: {peak_record.get('fan_speed', 'N/A')} RPM\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result
def get_specific_server_cpu_utilization(query: str) -> str:
    """
    Get CPU utilization data for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - CPU utilization request
    
    Returns:
        Formatted string with CPU utilization data for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    found_servers = []
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'CPU utilization for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} CPU utilization'")
    
    # Get CPU utilization data for all found servers
    results = []
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        
        # Check if CPU utilization data exists
        if not server_data.get("peak_cpu_record"):
            results.append({
                "serial": server_serial,
                "error": "No CPU utilization data available"
            })
            continue
        
        peak_cpu_record = server_data["peak_cpu_record"]
        
        # Get the peak CPU utilization from server rankings if available
        peak_cpu_util = None
        if server_rankings.get("top_cpu"):
            for serial, cpu_util in server_rankings["top_cpu"]:
                if serial == server_serial:
                    peak_cpu_util = cpu_util
                    break
        
        # Fallback to peak_cpu_record if not found in rankings
        if peak_cpu_util is None:
            peak_cpu_util = peak_cpu_record.get("cpu_util", "N/A")
        
        # Calculate efficiency rating
        avg_cpu = server_data.get("avg_cpu_util", 0)
        if avg_cpu == 0:
            efficiency = "idle"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
        
        # Get server ranking position if available
        ranking_position = "N/A"
        if server_rankings.get("top_cpu"):
            for i, (serial, _) in enumerate(server_rankings["top_cpu"]):
                if serial == server_serial:
                    ranking_position = f"#{i+1}"
                    break
        
        servers_with_data += 1
        
        results.append({
            "serial": server_serial,
            "peak_cpu_util": peak_cpu_util,
            "avg_cpu_util": avg_cpu,
            "timestamp": peak_cpu_record.get("time_str", "N/A"),
            "power_consumption": peak_cpu_record.get("power_consumption", "N/A"),
            "temperature": peak_cpu_record.get("temperature", "N/A"),
            "fan_speed": peak_cpu_record.get("fan_speed", "N/A"),
            "efficiency": efficiency,
            "ranking_position": ranking_position
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"CPU utilization data for server {result['serial']}:\n\n"
        output += f"   - Peak CPU Utilization: {result['peak_cpu_util']}%\n"
        output += f"   - Average CPU Utilization: {result['avg_cpu_util']}%\n"
        output += f"   - Timestamp of Peak: {result['timestamp']}\n"
        output += f"   - Power Consumption at Peak: {result['power_consumption']}W\n"
        output += f"   - Temperature at Peak: {result['temperature']}°C\n"
        output += f"   - Fan Speed at Peak: {result['fan_speed']} RPM\n"
        output += f"   - CPU Efficiency Rating: {result['efficiency'].capitalize()}\n"
        output += f"   - Fleet Ranking: {result['ranking_position']}\n"
        
    else:
        # Multiple servers output
        output = f"CPU utilization data for {len(found_servers)} specified servers:\n\n"
        
        if servers_with_data > 0:
            # Calculate summary statistics
            valid_peak_cpus = [r["peak_cpu_util"] for r in results if "error" not in r and r["peak_cpu_util"] is not None]
            valid_avg_cpus = [r["avg_cpu_util"] for r in results if "error" not in r and r["avg_cpu_util"] is not None]
            
            if valid_peak_cpus:
                highest_peak = max(valid_peak_cpus)
                lowest_peak = min(valid_peak_cpus)
                avg_peak = sum(valid_peak_cpus) / len(valid_peak_cpus)
                
                output += f"   - Highest Peak CPU Among Servers: {highest_peak}%\n"
                output += f"   - Lowest Peak CPU Among Servers: {lowest_peak}%\n"
                output += f"   - Average Peak CPU: {round(avg_peak, 2)}%\n"
            
            if valid_avg_cpus:
                overall_avg = sum(valid_avg_cpus) / len(valid_avg_cpus)
                output += f"   - Overall Average CPU Utilization: {round(overall_avg, 2)}%\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']} (Rank: {result['ranking_position']}):\n"
                output += f"   - Peak CPU: {result['peak_cpu_util']}%\n"
                output += f"   - Average CPU: {result['avg_cpu_util']}%\n"
                output += f"   - Peak Timestamp: {result['timestamp']}\n"
                output += f"   - Peak Power: {result['power_consumption']}W\n"
                output += f"   - Peak Temperature: {result['temperature']}°C\n"
                output += f"   - Peak Fan Speed: {result['fan_speed']} RPM\n"
                output += f"   - Efficiency: {result['efficiency'].capitalize()}\n\n"
    
    return output


def get_lowest_servers_by_cpu_util(query: str = "") -> str:
    if not server_rankings.get("bottom_cpu"):
        return "No CPU utilization data available for ranking lowest servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["bottom_cpu"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for lowest CPU utilization."
    result_header = "Server with lowest peak CPU utilization:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by lowest peak CPU utilization:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by lowest peak CPU utilization:\n\n"
    result = result_header
    for i, (serial, lowest_cpu) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("lowest_cpu_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        lowest_record = server_info["lowest_cpu_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Lowest CPU: {lowest_cpu}%\n"
        result += f"   - Timestamp: {lowest_record['time_str']}\n"
        result += f"   - Power: {lowest_record.get('power_consumption', 'N/A')}W\n"
        result += f"   - Temperature: {lowest_record.get('temperature', 'N/A')}°C\n"
        result += f"   - Fan Speed: {lowest_record.get('fan_speed', 'N/A')} RPM\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_top_servers_by_ambient_temp(query: str = "") -> str:
    if not server_rankings.get("top_amb_temp"):
        return "No ambient temperature data available for ranking top servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["top_amb_temp"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for top ambient temperature."
    result_header = "Server with highest ambient temperature:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by highest ambient temperature:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by highest ambient temperature:\n\n"
    result = result_header
    for i, (serial, max_temp) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("max_temp_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        temp_record = server_info["max_temp_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Highest Ambient Temperature: {max_temp}°C\n"
        result += f"   - Timestamp: {temp_record['time_str']}\n"
        result += f"   - CPU Utilization: {temp_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - CPU Power: {temp_record.get('cpu_watts', 'N/A')}W\n"
        result += f"   - DIMM Power: {temp_record.get('dimm_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_specific_server_ambient_temp(query: str) -> str:
    """
    Get ambient temperature data for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - Ambient temperature request
    
    Returns:
        Formatted string with ambient temperature data for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    found_servers = []
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'ambient temperature for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} ambient temperature'")
    
    # Get ambient temperature data for all found servers
    results = []
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        
        # Check if ambient temperature data exists
        if not server_data.get("max_temp_record"):
            results.append({
                "serial": server_serial,
                "error": "No ambient temperature data available"
            })
            continue
        
        temp_record = server_data["max_temp_record"]
        # Get the max ambient temperature from server rankings if available
        max_ambient_temp = None
        if server_rankings.get("top_amb_temp"):
            for serial, temp in server_rankings["top_amb_temp"]:
                if serial == server_serial:
                    max_ambient_temp = temp
                    break
        
        # Fallback to temp_record if not found in rankings
        if max_ambient_temp is None:
            max_ambient_temp = temp_record.get("amb_temp", "N/A")
        
        servers_with_data += 1
        
        results.append({
            "serial": server_serial,
            "max_ambient_temp": max_ambient_temp,
            "timestamp": temp_record.get("time_str", "N/A"),
            "cpu_util": temp_record.get("cpu_util", "N/A"),
            "cpu_watts": temp_record.get("cpu_watts", "N/A"),
            "dimm_watts": temp_record.get("dimm_watts", "N/A"),
            "avg_ambient_temp": server_data.get("avg_ambient_temp", "N/A")
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"Ambient temperature data for server {result['serial']}:\n\n"
        output += f"   - Highest Ambient Temperature: {result['max_ambient_temp']}°C\n"
        output += f"   - Timestamp of Peak: {result['timestamp']}\n"
        output += f"   - Average Ambient Temperature: {result['avg_ambient_temp']}°C\n"
        output += f"   - CPU Utilization at Peak: {result['cpu_util']}%\n"
        output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n"
        output += f"   - DIMM Power at Peak: {result['dimm_watts']}W\n"
        
    else:
        # Multiple servers output
        output = f"Ambient temperature data for {len(found_servers)} specified servers:\n\n"
        
        if servers_with_data > 0:
            # Calculate summary statistics
            valid_max_temps = [r["max_ambient_temp"] for r in results if "error" not in r and r["max_ambient_temp"] is not None]
            if valid_max_temps:
                avg_max_temp = sum(valid_max_temps) / len(valid_max_temps)
                highest_temp = max(valid_max_temps)
                lowest_temp = min(valid_max_temps)
                
                output += f"   - Highest Temperature Among Servers: {highest_temp}°C\n"
                output += f"   - Lowest Temperature Among Servers: {lowest_temp}°C\n"
                output += f"   - Average Maximum Temperature: {round(avg_max_temp, 2)}°C\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']}:\n"
                output += f"   - Highest Ambient Temp: {result['max_ambient_temp']}°C\n"
                output += f"   - Average Ambient Temp: {result['avg_ambient_temp']}°C\n"
                output += f"   - Peak Timestamp: {result['timestamp']}\n"
                output += f"   - CPU Util at Peak: {result['cpu_util']}%\n"
                output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n"
                output += f"   - DIMM Power at Peak: {result['dimm_watts']}W\n\n"
    
    return output

def get_lowest_servers_by_ambient_temp(query: str = "") -> str:
    if not server_rankings.get("bottom_amb_temp"):
        return "No ambient temperature data available for ranking lowest servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["bottom_amb_temp"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for lowest ambient temperature."
    result_header = "Server with lowest ambient temperature:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by lowest ambient temperature:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by lowest ambient temperature:\n\n"
    result = result_header
    for i, (serial, min_temp) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("min_temp_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        temp_record = server_info["min_temp_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Lowest Ambient Temperature: {min_temp}°C\n"
        result += f"   - Timestamp: {temp_record['time_str']}\n"
        result += f"   - CPU Utilization: {temp_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - CPU Power: {temp_record.get('cpu_watts', 'N/A')}W\n"
        result += f"   - DIMM Power: {temp_record.get('dimm_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_top_servers_by_peak(query: str = "") -> str:
    if not server_rankings.get("top_peak"):
        return "No peak data available for ranking top servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["top_peak"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for top peak values."
    result_header = "Server with highest peak value:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by highest peak value:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by highest peak value:\n\n"
    result = result_header
    for i, (serial, max_peak_val) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("max_peak_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        peak_record = server_info["max_peak_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Highest Peak Value: {max_peak_val}\n"
        result += f"   - Timestamp: {peak_record['time_str']}\n"
        result += f"   - CPU Utilization: {peak_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - Ambient Temperature: {peak_record.get('amb_temp', 'N/A')}°C\n"
        result += f"   - CPU Power: {peak_record.get('cpu_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_specific_server_peak_data(query: str) -> str:
    """
    Get peak data for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - Peak data request
    
    Returns:
        Formatted string with peak data for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    found_servers = []
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'peak data for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} peak values'")
    
    # Get peak data for all found servers
    results = []
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        
        # Check if peak data exists
        if not server_data.get("max_peak_record"):
            results.append({
                "serial": server_serial,
                "error": "No peak data available"
            })
            continue
        
        peak_record = server_data["max_peak_record"]
        # Get the max peak value from server rankings if available
        max_peak_value = None
        if server_rankings.get("top_peak"):
            for serial, peak_val in server_rankings["top_peak"]:
                if serial == server_serial:
                    max_peak_value = peak_val
                    break
        
        # Fallback to peak_record if not found in rankings
        if max_peak_value is None:
            max_peak_value = peak_record.get("peak_value", "N/A")
        
        servers_with_data += 1
        
        results.append({
            "serial": server_serial,
            "max_peak_value": max_peak_value,
            "timestamp": peak_record.get("time_str", "N/A"),
            "cpu_util": peak_record.get("cpu_util", "N/A"),
            "amb_temp": peak_record.get("amb_temp", "N/A"),
            "cpu_watts": peak_record.get("cpu_watts", "N/A"),
            "avg_peak_value": server_data.get("avg_peak_value", "N/A")
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"Peak data for server {result['serial']}:\n\n"
        output += f"   - Highest Peak Value: {result['max_peak_value']}\n"
        output += f"   - Timestamp of Peak: {result['timestamp']}\n"
        output += f"   - Average Peak Value: {result['avg_peak_value']}\n"
        output += f"   - CPU Utilization at Peak: {result['cpu_util']}%\n"
        output += f"   - Ambient Temperature at Peak: {result['amb_temp']}°C\n"
        output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n"
        
    else:
        # Multiple servers output
        output = f"Peak data for {len(found_servers)} specified servers:\n\n"
        
        if servers_with_data > 0:
            # Calculate summary statistics
            valid_peak_values = [r["max_peak_value"] for r in results if "error" not in r and r["max_peak_value"] is not None]
            if valid_peak_values:
                avg_peak = sum(valid_peak_values) / len(valid_peak_values)
                highest_peak = max(valid_peak_values)
                lowest_peak = min(valid_peak_values)
                
                output += f"   - Highest Peak Value Among Servers: {highest_peak}\n"
                output += f"   - Lowest Peak Value Among Servers: {lowest_peak}\n"
                output += f"   - Average Peak Value: {round(avg_peak, 2)}\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']}:\n"
                output += f"   - Highest Peak Value: {result['max_peak_value']}\n"
                output += f"   - Average Peak Value: {result['avg_peak_value']}\n"
                output += f"   - Peak Timestamp: {result['timestamp']}\n"
                output += f"   - CPU Util at Peak: {result['cpu_util']}%\n"
                output += f"   - Ambient Temp at Peak: {result['amb_temp']}°C\n"
                output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n\n"
    
    return output


def get_lowest_servers_by_peak(query: str = "") -> str:
    if not server_rankings.get("bottom_peak"):
        return "No peak data available for ranking lowest servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["bottom_peak"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for lowest peak values."
    result_header = "Server with lowest peak value:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by lowest peak value:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by lowest peak value:\n\n"
    result = result_header
    for i, (serial, min_peak_val) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("min_peak_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        peak_record = server_info["min_peak_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Lowest Peak Value: {min_peak_val}\n"
        result += f"   - Timestamp: {peak_record['time_str']}\n"
        result += f"   - CPU Utilization: {peak_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - Ambient Temperature: {peak_record.get('amb_temp', 'N/A')}°C\n"
        result += f"   - CPU Power: {peak_record.get('cpu_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def calculate_carbon_footprint(query: str) -> str:
    """
    Calculate carbon footprint for multiple servers or fleet-wide analysis.
    Handles requests for all servers or top N servers with different grid types.
    
    Args:
        query: Natural language query that may contain:
               - Carbon intensity type (e.g. "low carbon", "average", "high carbon")
               - Number of servers to show (e.g. "top 5", "all servers")
    
    Returns:
        Formatted string with carbon footprint results for multiple servers
    """
    if not processed_server_data:
        return "No server data available."
    
    # Default values
    carbon_intensity = 'average_grid'
    num_servers_to_show = extract_server_count(query, default=10)
    
    # Parse carbon intensity preference
    query_lower = query.lower()
    if "low carbon" in query_lower or "renewable" in query_lower:
        carbon_intensity = 'low_carbon_grid'
    elif "high carbon" in query_lower or "coal" in query_lower:
        carbon_intensity = 'high_carbon_grid'
    
    # Validate carbon intensity input
    if carbon_intensity not in DEFAULT_CARBON_INTENSITY:
        return f"Invalid carbon intensity. Choose from: {', '.join(DEFAULT_CARBON_INTENSITY.keys())}"
    
    intensity_factor = DEFAULT_CARBON_INTENSITY[carbon_intensity]
    total_co2 = 0.0
    results = []
    
    # Process all servers
    for serial in processed_server_data.keys():
        server_data = processed_server_data[serial]
        energy_kwh = server_data.get("estimated_energy_kwh", 0)
        
        # Skip servers with insufficient data
        if energy_kwh == 0:
            continue
        
        # Calculate CO2 emissions
        co2_kg = energy_kwh * intensity_factor
        total_co2 += co2_kg
        
        # Get efficiency rating
        avg_cpu = server_data["avg_cpu_util"]
        
        # Special case for 0% utilization
        if avg_cpu == 0:
            efficiency = "idle"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
        
        results.append({
            "serial": serial,
            "energy_kwh": round(energy_kwh, 2),
            "co2_kg": round(co2_kg, 2),
            "avg_cpu": avg_cpu,
            "efficiency": efficiency,
            "carbon_intensity": carbon_intensity
        })
    
    if not results:
        return "No valid server data available for carbon footprint calculation."
    
    # Multiple servers summary output
    grid_type_display = carbon_intensity.replace('_', ' ').title()
    available_servers = len(results)
    
    output = f"Carbon footprint summary for all {available_servers} servers ({grid_type_display}):\n\n"
    output += f"   - Total CO2 Emissions: {round(total_co2, 2)} kg\n"
    output += f"   - Average per Server: {round(total_co2/available_servers, 2)} kg\n\n"
    
    if num_servers_to_show == float('inf'):
        # Show all servers
        top_count = available_servers
        output += f"All {available_servers} servers:\n\n"
    else:
        # Show requested number of servers
        top_count = min(int(num_servers_to_show), available_servers)
        output += f"Top {top_count} highest emitting servers:\n\n"
    
    # Add top N highest emitters
    sorted_results = sorted(results, key=lambda x: x["co2_kg"], reverse=True)
    
    for i, res in enumerate(sorted_results[:top_count]):
        output += f"{i+1}. Server {res['serial']}:\n"
        output += f"   - CO2 Emissions: {res['co2_kg']} kg\n"
        output += f"   - Energy Consumed: {res['energy_kwh']} kWh\n"
        output += f"   - CPU Utilization: {res['avg_cpu']}%\n"
        output += f"   - Efficiency Rating: {res['efficiency'].capitalize()}\n\n"
            
    # Add efficiency distribution
    eff_dist = {}
    for res in results:
        eff_dist[res["efficiency"]] = eff_dist.get(res["efficiency"], 0) + 1
    
    output += "Energy efficiency distribution:\n\n"
    for eff, count in sorted(eff_dist.items()):
        percentage = round((count / available_servers) * 100, 1)
        output += f"   - {eff.capitalize()}: {count} servers ({percentage}%)\n"

    return output


def co2_emission_server(query: str) -> str:
    """
    Calculate carbon footprint for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - Optional carbon intensity type (e.g. "low carbon", "average", "high carbon")
    
    Returns:
        Formatted string with carbon footprint results for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    # Default carbon intensity
    carbon_intensity = 'average_grid'
    found_servers = []
    
    # Parse carbon intensity preference
    query_lower = query.lower()
    if "low carbon" in query_lower or "renewable" in query_lower:
        carbon_intensity = 'low_carbon_grid'
    elif "high carbon" in query_lower or "coal" in query_lower:
        carbon_intensity = 'high_carbon_grid'
    
    # Validate carbon intensity input
    if carbon_intensity not in DEFAULT_CARBON_INTENSITY:
        return f"Invalid carbon intensity. Choose from: {', '.join(DEFAULT_CARBON_INTENSITY.keys())}"
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'co2 emission for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} carbon footprint'")
    
    # Calculate CO2 emissions for all found servers
    intensity_factor = DEFAULT_CARBON_INTENSITY[carbon_intensity]
    results = []
    total_co2 = 0.0
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        energy_kwh = server_data.get("estimated_energy_kwh", 0)
        
        if energy_kwh == 0:
            results.append({
                "serial": server_serial,
                "error": "No energy consumption data available"
            })
            continue
        
        # Calculate CO2 emissions
        co2_kg = energy_kwh * intensity_factor
        total_co2 += co2_kg
        servers_with_data += 1
        
        # Get efficiency rating
        avg_cpu = server_data["avg_cpu_util"]
        
        # Special case for 0% utilization
        if avg_cpu == 0:
            efficiency = "idle"
            efficiency_note = "Server is idle (0% CPU utilization)"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
            efficiency_note = f"Energy Efficiency Rating: {efficiency.capitalize()}"
        
        results.append({
            "serial": server_serial,
            "energy_kwh": round(energy_kwh, 2),
            "co2_kg": round(co2_kg, 2),
            "avg_cpu": avg_cpu,
            "efficiency": efficiency,
            "efficiency_note": efficiency_note
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"Carbon footprint analysis for server {result['serial']}:\n\n"
        output += f"   - Energy Consumed: {result['energy_kwh']} kWh\n"
        output += f"   - CO2 Emissions: {result['co2_kg']} kg\n"
        output += f"   - Carbon Intensity: {carbon_intensity.replace('_', ' ').title()} ({intensity_factor} kg CO2/kWh)\n"
        output += f"   - Average CPU Utilization: {result['avg_cpu']}%\n"
        output += f"   - {result['efficiency_note']}\n"
        
    else:
        # Multiple servers output
        grid_type_display = carbon_intensity.replace('_', ' ').title()
        output = f"Carbon footprint analysis for {len(found_servers)} specified servers ({grid_type_display}):\n\n"
        
        if servers_with_data > 0:
            output += f"   - Total CO2 Emissions: {round(total_co2, 2)} kg\n"
            output += f"   - Average per Server: {round(total_co2/servers_with_data, 2)} kg\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']}:\n"
                output += f"   - CO2 Emissions: {result['co2_kg']} kg\n"
                output += f"   - Energy Consumed: {result['energy_kwh']} kWh\n"
                output += f"   - CPU Utilization: {result['avg_cpu']}%\n"
                output += f"   - Efficiency: {result['efficiency'].capitalize()}\n\n"
    
    return output

def calculate_carbon_footprint_lowest(query: str) -> str:
    """
    Calculate carbon footprint for servers based on natural language query.
    Can handle requests for specific servers or all servers with different grid types.
    Shows LOWEST CO2 emitting servers by default.
    
    Args:
        query: Natural language query that may contain:
               - Server serial number (e.g. "server ABC123")
               - Carbon intensity type (e.g. "low carbon", "average", "high carbon")
               - Number of servers to show (e.g. "top 5", "show 15")
    
    Returns:
        Formatted string with carbon footprint results (lowest emitters first)
    """
    # Default values
    server_serial = None
    carbon_intensity = 'average_grid'
    num_servers_to_show = extract_server_count(query, default=10)
    
    # Parse query for server serial
    query_lower = query.lower()
    if "server" in query_lower:
        # Extract potential serial number after "server"
        parts = query_lower.split("server")
        if len(parts) > 1:
            potential_serial = parts[1].strip().upper()
            if potential_serial in processed_server_data:
                server_serial = potential_serial
    
    # Parse carbon intensity preference
    if "low carbon" in query_lower or "renewable" in query_lower:
        carbon_intensity = 'low_carbon_grid'
    elif "high carbon" in query_lower or "coal" in query_lower:
        carbon_intensity = 'high_carbon_grid'
    
    # Validate carbon intensity input
    if carbon_intensity not in DEFAULT_CARBON_INTENSITY:
        return f"Invalid carbon intensity. Choose from: {', '.join(DEFAULT_CARBON_INTENSITY.keys())}"
    
    intensity_factor = DEFAULT_CARBON_INTENSITY[carbon_intensity]
    total_co2 = 0.0
    results = []
    
    # Determine which servers to process
    servers_to_process = [server_serial] if server_serial else processed_server_data.keys()
    
    for serial in servers_to_process:
        if serial not in processed_server_data:
            continue
            
        server_data = processed_server_data[serial]
        energy_kwh = server_data.get("estimated_energy_kwh", 0)
        
        # Skip servers with insufficient data
        if energy_kwh == 0:
            continue
        
        # Calculate CO2 emissions
        co2_kg = energy_kwh * intensity_factor
        total_co2 += co2_kg
        
        # Get efficiency rating
        avg_cpu = server_data["avg_cpu_util"]
        
        # Special case for 0% utilization
        if avg_cpu == 0:
            efficiency = "idle"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
        
        results.append({
            "serial": serial,
            "energy_kwh": round(energy_kwh, 2),
            "co2_kg": round(co2_kg, 2),
            "avg_cpu": avg_cpu,
            "efficiency": efficiency,
            "carbon_intensity": carbon_intensity
        })
    
    if not results:
        return "No valid server data available for carbon footprint calculation."
    
    # Format the results
    if server_serial:
        # Single server detailed output
        result = next((r for r in results if r["serial"] == server_serial), None)
        if not result:
            return f"No data available for server {server_serial}."
        
        output = f"Carbon footprint analysis for server {server_serial}:\n\n"
        output += f"   - Energy Consumed: {result['energy_kwh']} kWh\n"
        output += f"   - CO2 Emissions: {result['co2_kg']} kg\n"
        output += f"   - Carbon Intensity: {carbon_intensity.replace('_', ' ').title()} ({intensity_factor} kg CO2/kWh)\n"
        output += f"   - Average CPU Utilization: {result['avg_cpu']}%\n"
        
        if result['efficiency'] == 'idle':
            output += "   - Energy Efficiency: Server is idle (0% CPU utilization)\n"
        else:
            output += f"   - Energy Efficiency Rating: {result['efficiency'].capitalize()}\n"
            
        return output
    
    else:
        # Multiple servers summary output - LOWEST emitters
        grid_type_display = carbon_intensity.replace('_', ' ').title()
        available_servers = len(results)
        
        output = f"Carbon footprint summary for all {available_servers} servers ({grid_type_display}):\n\n"
        output += f"   - Total CO2 Emissions: {round(total_co2, 2)} kg\n"
        output += f"   - Average per Server: {round(total_co2/available_servers, 2)} kg\n\n"
        
        if num_servers_to_show == float('inf'):
            # Show all servers
            top_count = available_servers
            output += f"All {available_servers} servers (lowest to highest emissions):\n\n"
        else:
            # Show requested number of servers
            top_count = min(int(num_servers_to_show), available_servers)
            output += f"Top {top_count} LOWEST emitting servers:\n\n"
        
        # Sort by LOWEST emissions (ascending order)
        sorted_results = sorted(results, key=lambda x: x["co2_kg"], reverse=False)
        
        for i, res in enumerate(sorted_results[:top_count]):
            output += f"{i+1}. Server {res['serial']}:\n"
            output += f"   - CO2 Emissions: {res['co2_kg']} kg\n"
            output += f"   - Energy Consumed: {res['energy_kwh']} kWh\n"
            output += f"   - CPU Utilization: {res['avg_cpu']}%\n"
            output += f"   - Efficiency Rating: {res['efficiency'].capitalize()}\n\n"
                
        # Add efficiency distribution
        eff_dist = {}
        for res in results:
            eff_dist[res["efficiency"]] = eff_dist.get(res["efficiency"], 0) + 1
        
        output += "Energy efficiency distribution:\n\n"
        for eff, count in sorted(eff_dist.items()):
            percentage = round((count / available_servers) * 100, 1)
            output += f"   - {eff.capitalize()}: {count} servers ({percentage}%)\n"
    
    return output

def get_server_stats(query: str) -> str:
    specific_server_serial = None
    match = re.search(r"server\s+([A-Z0-9-]+)", query, re.IGNORECASE)
    if match:
        potential_serial = match.group(1).upper()
        if potential_serial in processed_server_data:
            specific_server_serial = potential_serial
    if specific_server_serial:
        data = processed_server_data[specific_server_serial]
        result = f"Server {specific_server_serial} Statistics:\n" + "=" * (len(specific_server_serial) + 20) + "\n\n"
        latest = data.get("latest_record")
        if latest:
            result += f"Latest Observation ({latest.get('time_str', 'N/A')}):\n"
            if latest.get('cpu_util') is not None: result += f"  CPU Utilization: {latest['cpu_util']}%\n"
            if latest.get('amb_temp') is not None: result += f"  Ambient Temperature: {latest['amb_temp']}°C\n"
            if latest.get('peak') is not None: result += f"  Peak Value: {latest['peak']}\n"
            if latest.get('estimated_power') is not None: result += f"  Estimated Power: {latest['estimated_power']}W\n"
        else:
            result += "No latest observation data available.\n"
        result += "\nSummary Metrics:\n"
        if data.get('avg_cpu_util') is not None: result += f"  Average CPU Utilization: {data['avg_cpu_util']}%\n"
        peak_cpu_rec = data.get("peak_cpu_record")
        if peak_cpu_rec and peak_cpu_rec.get('cpu_util') is not None:
            result += f"  Peak CPU: {peak_cpu_rec['cpu_util']}% at {peak_cpu_rec.get('time_str', 'N/A')}\n"
        lowest_cpu_rec = data.get("lowest_cpu_record")
        if lowest_cpu_rec and lowest_cpu_rec.get('cpu_util') is not None:
            result += f"  Lowest CPU: {lowest_cpu_rec['cpu_util']}% at {lowest_cpu_rec.get('time_str', 'N/A')}\n"
        if data.get('max_amb_temp') is not None:
            max_temp_rec = data.get("max_temp_record")
            result += f"  Max Ambient Temp: {data['max_amb_temp']}°C at {max_temp_rec.get('time_str', 'N/A') if max_temp_rec else 'N/A'}\n"
        if data.get('min_amb_temp') is not None:
            min_temp_rec = data.get("min_temp_record")
            result += f"  Min Ambient Temp: {data['min_amb_temp']}°C at {min_temp_rec.get('time_str', 'N/A') if min_temp_rec else 'N/A'}\n"
        if data.get('estimated_energy_kwh') is not None:
            result += f"  Total Estimated Energy: {data['estimated_energy_kwh']} kWh\n"
        if data.get('co2_emissions'):
            result += f"  Est. CO2 (avg grid): {data['co2_emissions'].get('average_grid', 'N/A')} kg\n"
        return result.strip()
    result = "Server Fleet Statistics Summary (Top 10 by Peak CPU shown):\n" + "=" * 50 + "\n\n"
    if not processed_server_data: return "No server data available for summary."
    servers_to_show = server_rankings.get("top_cpu", [])[:10]
    if not servers_to_show:
        result += "No ranked servers to display in summary.\n"
        count = 0
        for serial, data_item in processed_server_data.items():
            if count >=5: break
            latest = data_item.get("latest_record", {})
            result += f"Server {serial} (Last Seen: {latest.get('time_str', 'N/A')}):\n"
            result += f"  Latest CPU: {latest.get('cpu_util', 'N/A')}%, Peak CPU: {data_item.get('peak_cpu_util', 'N/A')}%\n"
            result += f"  Latest Amb Temp: {latest.get('amb_temp', 'N/A')}°C, Max Amb Temp: {data_item.get('max_amb_temp', 'N/A')}°C\n\n"
            count +=1
        if count == 0: result += "No server data processed to display.\n"
    else:
        for serial, peak_cpu in servers_to_show:
            data = processed_server_data[serial]
            latest = data.get("latest_record", {})
            result += f"Server {serial} (Peak CPU: {peak_cpu}%):\n"
            result += f"  Last Observed: {latest.get('time_str', 'N/A')}\n"
            result += f"  Current CPU: {latest.get('cpu_util', 'N/A')}%\n"
            result += f"  Current Ambient: {latest.get('amb_temp', 'N/A')}°C\n\n"
    result += "For specific server details, query 'stats for server [SERIAL_NUMBER]'.\n"
    result += f"Total processed servers: {len(processed_server_data)}.\n"
    return result

def get_server_timestamps(query: str) -> str:
    if not processed_server_data: return "No server data available."
    server_patterns = [r'server\s+([A-Za-z0-9_-]+)', r'([A-Za-z0-9_-]{5,})']
    server_serial = None
    for pattern in server_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            potential_serial = match.group(1).upper()
            if potential_serial in processed_server_data:
                server_serial = potential_serial
                break
    if not server_serial:
        query_upper = query.upper()
        for serial_key in processed_server_data.keys():
            if serial_key in query_upper:
                server_serial = serial_key
                break
    if not server_serial:
        available_servers = list(processed_server_data.keys())
        return f"Could not identify server. Examples: {', '.join(available_servers[:3])}{'...' if len(available_servers) > 3 else ''}"
    server_info = processed_server_data[server_serial]
    if 'all_records' not in server_info or not server_info['all_records']:
        return f"No timestamp data for server {server_serial}"
    timestamps = [record['time_str'] for record in server_info['all_records'] if 'time_str' in record]
    if not timestamps: return f"No timestamps found for server {server_serial}"
    result = f"Timestamps for server {server_serial}:\nTotal records: {len(timestamps)}\n\n"
    display_count = min(20, len(timestamps))
    for i, timestamp in enumerate(timestamps[:display_count], 1):
        result += f"{i:2d}. {timestamp}\n"
    if len(timestamps) > display_count:
        result += f"\n... and {len(timestamps) - display_count} more timestamps"
    return result

def identify_high_cpu_servers(query: str) -> str:
    if not processed_server_data: return "No server data available."
    match = re.search(r'(\d+(\.\d+)?)', query)
    if not match: return "Please specify a CPU threshold (e.g., 'CPU above 80%')"
    try: threshold = float(match.group(1))
    except ValueError: return "Invalid number for CPU threshold."
    if not (0 <= threshold <= 100): return f"Invalid threshold: {threshold}%. Must be 0-100."
    high_cpu_servers_details = []
    for serial, server_info in processed_server_data.items():
        if 'all_records' not in server_info or not server_info['all_records']: continue
        high_cpu_count = 0
        max_cpu_this_server = 0.0
        for record in server_info['all_records']:
            cpu_util = record.get('cpu_util')
            if cpu_util is None: continue
            try: cpu_util_float = float(cpu_util)
            except (ValueError, TypeError): continue
            if cpu_util_float > threshold: high_cpu_count += 1
            if cpu_util_float > max_cpu_this_server: max_cpu_this_server = cpu_util_float
        if high_cpu_count > 0:
            total_records = len(server_info['all_records'])
            percentage_high_cpu_time = (high_cpu_count / total_records) * 100 if total_records > 0 else 0
            high_cpu_servers_details.append({'serial': serial, 'high_cpu_count': high_cpu_count,
                                             'total_records': total_records, 'percentage': percentage_high_cpu_time,
                                             'max_cpu_observed': max_cpu_this_server})
    if not high_cpu_servers_details: return f"No servers found with CPU above {threshold}%"
    high_cpu_servers_details.sort(key=lambda x: (x['percentage'], x['max_cpu_observed']), reverse=True)
    result = f"Servers with CPU records above {threshold}% (sorted by prevalence & max CPU):\n"
    result += f"Found {len(high_cpu_servers_details)} server(s) out of {len(processed_server_data)} total.\n\n"
    for i, stats in enumerate(high_cpu_servers_details[:20], 1):
        result += f"{i:2d}. Server: {stats['serial']}\n"
        result += f"    Instances >{threshold}%: {stats['high_cpu_count']}/{stats['total_records']} ({stats['percentage']:.1f}% of records)\n"
        result += f"    Highest CPU recorded: {stats['max_cpu_observed']:.1f}%\n\n"
    if len(high_cpu_servers_details) > 20: result += f"... and {len(high_cpu_servers_details) - 20} more."
    return result

def get_ambient_temp_stats(query: str) -> str:
    if not processed_server_data: return "Error: No server data for ambient temp stats."
    query_lower = query.lower()
    serial_match = re.search(r"server\s+([A-Z0-9-]+)", query_lower, re.IGNORECASE)
    specific_serial = None
    if serial_match:
        potential_serial = serial_match.group(1).upper()
        if potential_serial in processed_server_data: specific_serial = potential_serial
        else: return f"Server {potential_serial} not found for temp stats."
    if specific_serial:
        data = processed_server_data[specific_serial]
        result = f"Ambient Temp Stats for Server {specific_serial}:\n" + "=" * (len(specific_serial) + 30) + "\n\n"
        latest = data.get("latest_record", {})
        max_rec = data.get("max_temp_record", {})
        min_rec = data.get("min_temp_record", {})
        if latest.get('amb_temp') is not None: result += f"  Current: {latest['amb_temp']}°C (at {latest.get('time_str', 'N/A')})\n"
        else: result += "  Current: N/A\n"
        if data.get('max_amb_temp') is not None: result += f"  Maximum: {data['max_amb_temp']}°C (at {max_rec.get('time_str', 'N/A') if max_rec else 'N/A'})\n"
        else: result += "  Maximum: N/A\n"
        if data.get('min_amb_temp') is not None: result += f"  Minimum: {data['min_amb_temp']}°C (at {min_rec.get('time_str', 'N/A') if min_rec else 'N/A'})\n"
        else: result += "  Minimum: N/A\n"
        if data.get('max_amb_temp') is not None and data.get('min_amb_temp') is not None:
            result += f"  Range: {data['max_amb_temp'] - data['min_amb_temp']:.1f}°C\n"
        else: result += "  Range: N/A\n"
        all_server_temps = [rec['amb_temp'] for rec in data.get("all_records", []) if rec.get('amb_temp') is not None]
        if all_server_temps:
            avg_server_temp = sum(all_server_temps) / len(all_server_temps)
            result += f"  Average: {avg_server_temp:.1f}°C (over all its records)\n"
        else: result += "  Average: N/A\n"
        return result.strip()
    result = "Overall Ambient Temperature Statistics (Fleet):\n" + "=" * 45 + "\n\n"
    if server_rankings["top_amb_temp"]:
        result += "🏆 Top 5 Highest Max Ambient Temperatures:\n"
        for i, (serial, temp) in enumerate(server_rankings["top_amb_temp"][:5], 1):
            result += f"  {i}. Server {serial}: {temp}°C (at {processed_server_data[serial]['max_temp_record']['time_str']})\n"
        result += "\n"
    if server_rankings["bottom_amb_temp"]:
        result += "❄️ Top 5 Lowest Min Ambient Temperatures:\n"
        for i, (serial, temp) in enumerate(server_rankings["bottom_amb_temp"][:5], 1):
            result += f"  {i}. Server {serial}: {temp}°C (at {processed_server_data[serial]['min_temp_record']['time_str']})\n"
        result += "\n"
    all_latest_temps = [s_data['latest_record']['amb_temp'] for s_data in processed_server_data.values()
                        if s_data.get('latest_record') and s_data['latest_record'].get('amb_temp') is not None]
    if all_latest_temps:
        avg_fleet_temp = sum(all_latest_temps) / len(all_latest_temps)
        max_fleet_latest_temp = max(all_latest_temps)
        min_fleet_latest_temp = min(all_latest_temps)
        result += f"🌡️ Current Fleet Ambient Temperatures (latest records):\n"
        result += f"   - Average: {avg_fleet_temp:.1f}°C\n"
        result += f"   - Highest Current: {max_fleet_latest_temp:.1f}°C\n"
        result += f"   - Lowest Current: {min_fleet_latest_temp:.1f}°C\n"
    return result.strip()


def get_filtered_server_records(query_params_str: str) -> str:
    try:
        params = json.loads(query_params_str)
        server_serial, metric_key, operator, value = params.get("server_serial"), params.get("metric"), params.get("operator"), params.get("value")
        if not all([server_serial, metric_key, operator, value is not None]):
            return "Error: Missing one or more required JSON fields: 'server_serial', 'metric', 'operator', 'value'."
        server_serial = server_serial.upper()
        if server_serial not in processed_server_data: return f"Error: Server {server_serial} not found."
        server_info = processed_server_data[server_serial]
        if 'all_records' not in server_info or not server_info['all_records']:
            return f"No records for server {server_serial}."
        if metric_key not in ['cpu_util', 'amb_temp', 'peak']:
            return f"Error: Unsupported metric '{metric_key}'. Use: cpu_util, amb_temp, peak."
        if operator not in ['greater_than', 'less_than', 'equals']:
            return f"Error: Unsupported operator '{operator}'. Use: greater_than, less_than, equals."
        try: filter_value = float(value)
        except ValueError: return f"Error: Filter value '{value}' must be numeric."
        matching_records_info = []
        for record in server_info['all_records']:
            record_value = record.get(metric_key)
            if record_value is None: continue
            try: record_value_float = float(record_value)
            except (ValueError, TypeError): continue
            match = False
            if operator == 'greater_than' and record_value_float > filter_value: match = True
            elif operator == 'less_than' and record_value_float < filter_value: match = True
            elif operator == 'equals' and record_value_float == filter_value: match = True
            if match:
                matching_records_info.append(f"- Timestamp: {record['time_str']}, {metric_key.replace('_', ' ').title()}: {record_value}")
        if not matching_records_info:
            return f"No records for {server_serial} where {metric_key} {operator.replace('_',' ')} {filter_value}."
        result = f"Filtered records for {server_serial} ({metric_key} {operator.replace('_',' ')} {filter_value}):\n"
        result += f"Found {len(matching_records_info)} record(s).\n\n"
        display_count = min(20, len(matching_records_info))
        for i, rec_info in enumerate(matching_records_info[:display_count], 1):
            result += f"{i:2d}. {rec_info}\n"
        if len(matching_records_info) > display_count:
            result += f"\n... and {len(matching_records_info) - display_count} more."
        return result
    except json.JSONDecodeError:
        return "Error: Invalid JSON for Action Input. E.g., '{\"server_serial\": \"XYZ\", \"metric\": \"cpu_util\", \"operator\": \"greater_than\", \"value\": 10}'. Double quotes essential."
    except Exception as e:
        logger.error(f"Error in get_filtered_server_records: {e}", exc_info=True)
        return f"Unexpected error filtering records: {str(e)}"

def extract_server_name(query: str, all_servers: set) -> Optional[str]:
    """Extracts a likely server name from the query based on known server IDs."""
    upper_query = query.upper()

    # Match all uppercase-alphanumeric-underscore-hyphen patterns
    candidates = re.findall(r'\b[A-Z0-9_\\-]{3,}\b', upper_query)
    for candidate in candidates:
        if candidate in all_servers:
            return candidate
    return None

def detect_anomalies(query: str) -> str:
    METRIC_KEYWORDS = {
    "cpu_util": ["cpu utilization", "cpu util", "cpu usage", "strange behavior in cpu", "cpu load"],
    "amb_temp": ["ambient temperature", "amb temp", "temperature", "temperature spikes"],
    "cpu_watts": ["cpu power", "cpu watts", "power consumption", "power usage"],
    "dimm_watts": ["memory power", "dimm watts", "dimm memory power"],
}
    
    def extract_metrics(query: str) -> List[str]:
        query = query.lower()
        matched = []
        for metric, aliases in METRIC_KEYWORDS.items():
            for phrase in aliases:
                if phrase in query:
                    matched.append(metric)
                    break
        return matched if matched else ["cpu_util", "amb_temp", "cpu_watts", "dimm_watts"]


    # Parse query to determine scope
    analyze_all = True
    specific_server = None
    specific_metric = None

    query_lower = query.lower()
    
    potential_serial = extract_server_name(query, set(processed_server_data.keys()))
    if potential_serial:
        analyze_all = False
        specific_server = potential_serial
    elif re.search(r"\bserver\b", query.lower()):
        return f"⚠️ Server mentioned but not found in dataset. Please check the name."
    # otherwise, proceed with analyze_all = True



    
    # Determine which metric to analyze
    metrics_to_check = extract_metrics(query)
    
    # Check for multi-word metrics first
    if "cpu watts" in query_lower or "cpu_watts" in query_lower:
        metrics_to_check = ["cpu_watts"]
    elif "dimm watts" in query_lower or "dimm_watts" in query_lower:
        metrics_to_check = ["dimm_watts"]
    elif "cpu util" in query_lower or "cpu_util" in query_lower:
        metrics_to_check = ["cpu_util"]
    elif "amb temp" in query_lower or "amb_temp" in query_lower:
        metrics_to_check = ["amb_temp"]
    # Fallback to single-word matches
    elif "watts" in query_lower:
        metrics_to_check = ["cpu_watts", "dimm_watts"]
    elif "temp" in query_lower:
        metrics_to_check = ["amb_temp"]
    elif "cpu" in query_lower or "util" in query_lower:
        metrics_to_check = ["cpu_util"]
    
    if not metrics_to_check:
        metrics_to_check = ["cpu_util", "amb_temp", "cpu_watts", "dimm_watts"]

    
    # Enhanced statistical anomaly detection
    def find_anomalies(values, timestamps, metric_name, server_serial):
        if not values or len(values) < 3:
            return [], None
        
        values = [float(v) for v in values]
        median = sorted(values)[len(values)//2]
        deviations = [abs(x - median) for x in values]
        mad = sorted(deviations)[len(deviations)//2]
        
        if mad == 0:
            return [], median
        
        base_threshold = 3.5  # Lower base threshold
        threshold = base_threshold
        
        # More sensitive scaling
        if len(values) < 10:
            threshold = 3.0  # Very sensitive for tiny datasets
        elif len(values) > 100:
            threshold = base_threshold + (len(values) / 500)  # Faster scaling
        
        # Debug output (optional)
        # print(f"Debug - {metric_name}: Median={median}, MAD={mad}, Threshold={threshold}")
        # print(f"Values: {values}, Z-scores: {modified_z_scores}")
        
        # Find anomalies with deduplication
        anomalies = []
        seen = set()
        modified_z_scores = [0.6745 * (x - median) / mad for x in values]
        
        for i, score in enumerate(modified_z_scores):
            if abs(score) > threshold:
                anomaly_key = f"{values[i]}-{timestamps[i]}"
                if anomaly_key not in seen:
                    seen.add(anomaly_key)
                    anomalies.append({
                        "value": values[i],
                        "z_score": round(score, 2),
                        "metric": metric_name,
                        "server": server_serial,
                        "timestamp": timestamps[i]
                    })
        
        return anomalies, median
    
    # Process requested servers
    servers_to_check = list(processed_server_data.keys()) if analyze_all else [specific_server]
    all_anomalies = []
    median_baselines = {}
    
    for serial in servers_to_check:
        server_data = processed_server_data.get(serial)
        if not server_data:
            continue
            
        records = server_data.get("all_records", [])
        if not records:
            continue
            
        # Check each requested metric
        for metric in metrics_to_check:
            values = []
            timestamps = []
            
            for record in records:
                val = record.get(metric)
                if val is not None:
                    values.append(val)
                    timestamps.append(record["time_str"])
            
            if values:
                metric_anomalies, median = find_anomalies(values, timestamps, metric, serial)
                if metric not in median_baselines:
                    median_baselines[metric] = median
                all_anomalies.extend(metric_anomalies)
    
    # Format results with enhanced information
    if not all_anomalies:
        if analyze_all:
            return "No significant anomalies detected across all servers and metrics."
        return f"No significant anomalies detected for server {specific_server}."
    
    # Group by severity
    critical = [a for a in all_anomalies if abs(a["z_score"]) > 5]
    major = [a for a in all_anomalies if 3.5 < abs(a["z_score"]) <= 5]
    
    # Temporal analysis
    anomaly_hours = [a['timestamp'].split(', ')[1][:2] for a in all_anomalies]
    hour_dist = Counter(anomaly_hours).most_common(3)
    
    # Build enhanced output
    output = []
    if analyze_all:
        output.append(f"📊 Enhanced Anomaly Report for {len(servers_to_check)} servers")
    else:
        output.append(f"📊 Enhanced Anomaly Report for server {specific_server}")
    
    # Baseline context
    output.append("\n🔍 Normal Ranges (median values):")
    for metric, median in median_baselines.items():
        output.append(f"- {metric}: {median}")
    
    # Critical anomalies
    if critical:
        output.append("\n🚨 CRITICAL ANOMALIES (z-score > 5):")
        for a in critical[:5]:  # Top 5 only
            output.append(
                f"- {a['server']} | {a['metric']} = {a['value']} "
                f"(z-score: {a['z_score']}) at {a['timestamp']}"
            )
    
    # Major anomalies
    if major:
        output.append("\n⚠️ MAJOR ANOMALIES (3.5 < z-score ≤ 5):")
        for a in major[:5]:  # Top 5 only
            output.append(
                f"- {a['server']} | {a['metric']} = {a['value']} "
                f"(z-score: {a['z_score']}) at {a['timestamp']}"
            )
    
    # Temporal patterns
    if hour_dist:
        output.append("\n⏰ Frequent Anomaly Times:")
        for hour, count in hour_dist:
            output.append(f"- {hour}:00 - {count} anomalies")
    
    # Root cause suggestions
    output.append("\n🔧 Potential Investigation Paths:")
    if 'cpu_watts' in median_baselines:
        output.append("- CPU Power Spikes: Check workload scheduler and cooling")
    if 'amb_temp' in median_baselines:
        output.append("- Temp Fluctuations: Verify HVAC and rack airflow")
    if 'dimm_watts' in median_baselines:
        output.append("- Memory Power: Run DIMM diagnostics")
    
    # Summary
    total_anomalies = len(critical) + len(major)
    output.append(f"\n📈 Found {total_anomalies} significant anomalies (showing top 5 each)")
    
    return "\n".join(output)


def load_pdf_documents(base_path: str = BASE_PATH) -> List:
    """Load PDF documents from the specified base path"""
    logger.info(f"Loading PDF documents from {base_path}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(base_path):
        logger.warning(f"Document path {base_path} does not exist. Creating directory.")
        os.makedirs(base_path, exist_ok=True)
        return []
    
    entries = glob.glob(os.path.join(base_path, "*"))
    documents = []

    for entry in entries:
        if os.path.isdir(entry):
            pdf_files = glob.glob(os.path.join(entry, "**", "*.pdf"), recursive=True)
        elif entry.lower().endswith(".pdf"):
            pdf_files = [entry]
        else:
            continue

        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                
                # Add better metadata handling
                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(pdf_file)
                    doc.metadata["file_path"] = pdf_file
                    doc.metadata["file_size"] = os.path.getsize(pdf_file)
                    documents.append(doc)
                    
                logger.info(f"Loaded PDF: {pdf_file} ({len(docs)} pages)")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                continue  # Continue with other files instead of breaking
    
    logger.info(f"Loaded {len(documents)} document chunks from {len(set(doc.metadata['source_file'] for doc in documents))} files")
    return documents

def setup_vector_store(documents: List) -> Optional[object]:
    """Set up vector store with the provided documents"""
    if not documents:
        logger.warning("No documents provided to setup_vector_store")
        return None
        
    logger.info("Setting up vector store")
    
    # Ensure chroma_db directory exists
    chroma_db_path = CHROMA_DB_PATH
    if not os.path.exists(chroma_db_path):
        logger.info(f"Creating ChromaDB directory: {chroma_db_path}")
        os.makedirs(chroma_db_path, exist_ok=True)
    
    # Use RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]  # Better separation strategy
    )
    
    try:
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(split_docs)} chunks")
        
        # Initialize embeddings with error handling
        embeddings = Embeddings
        
        # Create vector store with better error handling
        try:
            # Check if existing ChromaDB exists and try to load it first
            if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
                try:
                    logger.info("Attempting to load existing ChromaDB...")
                    vectorstore = Chroma(
                        persist_directory=chroma_db_path,
                        embedding_function=embeddings
                    )
                    # Test if the vectorstore works
                    test_results = vectorstore.similarity_search("test", k=1)
                    logger.info("Successfully loaded existing ChromaDB")
                    return vectorstore
                except Exception as load_error:
                    logger.warning(f"Could not load existing ChromaDB: {load_error}. Creating new one.")
            
            # Create new vector store
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=chroma_db_path
            )
            vectorstore.persist()  # Explicitly persist
            logger.info("Persistent vector store created successfully")
            
        except Exception as persist_error:
            logger.warning(f"Error creating persistent vector store: {persist_error}")
            logger.info("Falling back to in-memory vector store")
            vectorstore = Chroma.from_documents(
                documents=split_docs, 
                embedding=embeddings
            )
            logger.info("In-memory vector store created")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error in setup_vector_store: {e}")
        return None

def setup_rag_chain(vectorstore: object, llm_instance: object) -> Optional[object]:
    """Set up and return the RAG conversation chain with memory"""
    if not vectorstore:
        logger.error("No vectorstore provided to setup_rag_chain")
        return None
        
    if not llm_instance:
        logger.error("No LLM instance provided to setup_rag_chain")
        return None
        
    try:
        # Set up the retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # Explicitly set search type
            search_kwargs={"k": 3}
        )
        
        # Set up conversation buffer memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Create the conversational retrieval chain with memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=LLM_INSTANCE,
            retriever=retriever,
            chain_type="stuff",
            memory=memory,
            return_source_documents=True,
            verbose=False,
            max_tokens_limit=4000  # Add token limit to prevent context overflow
        )
        
        logger.info("RAG chain with conversation memory created successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {e}")
        return None

# -----------------------------
# RAG Tool Functions
# -----------------------------
# Global RAG variables
rag_chain: Optional[object] = None
vectorstore: Optional[object] = None

def query_documents(query: str) -> str:
    """Query the document knowledge base using RAG"""
    global rag_chain
    
    if not rag_chain:
        return "Document search is not available. No documents have been loaded. Please initialize the RAG system first."
    
    if not query or not query.strip():
        return "Please provide a specific question to search the documents."
    
    try:
        # Add chat history parameter (empty for stateless queries)
        response = rag_chain({
            "question": query.strip(),
            "chat_history": []  # Required parameter for ConversationalRetrievalChain
        })
        
        answer = response.get("answer", "No answer found.")
        source_docs = response.get("source_documents", [])
        
        formatted_response = f"**Answer:** {answer}\n\n"
        
        if source_docs:
            formatted_response += "**Sources:**\n"
            seen_sources = set()
            for doc in source_docs[:3]:  # Limit to top 3 sources
                source_file = doc.metadata.get("source_file", "Unknown")
                if source_file not in seen_sources:
                    formatted_response += f"- {source_file}\n"
                    seen_sources.add(source_file)
        else:
            formatted_response += "**Sources:** No relevant sources found.\n"
        
        return formatted_response.strip()
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return f"Error occurred while searching documents: {str(e)}"

def list_available_documents() -> str:
    """List all available documents in the knowledge base"""
    global vectorstore
    
    if not vectorstore:
        return "No documents are currently loaded in the knowledge base. Please initialize the RAG system first."
    
    try:
        all_docs = vectorstore.get()
        
        if not all_docs or not all_docs.get('metadatas'):
            return "No documents found in the knowledge base."
        
        source_files = set()
        total_chunks = 0
        
        for metadata in all_docs['metadatas']:
            if metadata and isinstance(metadata, dict):  # Add type check
                source_file = metadata.get('source_file', 'Unknown')
                source_files.add(source_file)
                total_chunks += 1
        
        if not source_files:
            return "No valid documents found in the knowledge base."
        
        response = f"**Available Documents ({len(source_files)} files):**\n"
        for source_file in sorted(source_files):
            response += f"- {source_file}\n"
        
        response += f"\n**Total document chunks:** {total_chunks}"
        return response
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return "Error occurred while listing available documents."

# -----------------------------
# Initialize RAG System
# -----------------------------
def initialize_rag_system(llm_instance: object, documents_path: str = "C:\\Users\\sahil\\Desktop\\chat\\pdf") -> bool:
    """
    Initialize RAG system - call this in your setup_agent function
    Returns True if successful, False otherwise
    """
    global rag_chain, vectorstore
    
    logger.info("Initializing RAG system...")
    
    if not llm_instance:
        logger.error("No LLM instance provided to initialize_rag_system")
        return False
    
    try:
        # Check if we already have a vectorstore loaded
        if vectorstore and rag_chain:
            logger.info("RAG system already initialized")
            return True
            
        documents = load_pdf_documents(documents_path)
        
        if not documents:
            logger.warning("No documents found - RAG system will be unavailable")
            return False
            
        vectorstore = setup_vector_store(documents)
        if not vectorstore:
            logger.error("Vector store setup failed")
            return False
            
        rag_chain = setup_rag_chain(vectorstore, llm_instance)
        if not rag_chain:
            logger.error("RAG chain setup failed")
            return False
            
        logger.info("RAG system initialized successfully")
        return True
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        rag_chain = None
        vectorstore = None
        return False

def auto_initialize_rag_on_startup(llm_instance: object) -> bool:
    """
    Automatically initialize RAG system on startup
    This function will be called during system initialization
    """
    logger.info("Starting automatic RAG system initialization...")
    
    # List of possible document directories to check
    possible_paths = [
        "./pdf",
         BASE_PATH
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found document directory: {path}")
            success = initialize_rag_system(llm_instance, path)
            if success:
                return True
        else:
            logger.info(f"Document directory not found: {path}")
    
    # If no existing directories found, create default one
    default_path = "./documents"
    logger.info(f"Creating default documents directory: {default_path}")
    os.makedirs(default_path, exist_ok=True)
    
    # Try to initialize with empty directory (will result in no documents but system ready)
    success = initialize_rag_system(llm_instance, default_path)
    
    if success:
        logger.info("RAG system initialized with empty document directory")
    else:
        logger.warning("RAG system initialization failed even with empty directory")
    
    return success

# -----------------------------
# Utility Functions
# -----------------------------
def reset_rag_system() -> None:
    """Reset the RAG system (useful for reloading documents)"""
    global rag_chain, vectorstore
    rag_chain = None
    vectorstore = None
    logger.info("RAG system reset")

def get_rag_system_status() -> dict:
    """Get the current status of the RAG system"""
    global rag_chain, vectorstore
    
    return {
        "rag_chain_initialized": rag_chain is not None,
        "vectorstore_initialized": vectorstore is not None,
        "system_ready": rag_chain is not None and vectorstore is not None,
        "chroma_db_exists": os.path.exists("./chroma_db"),
        "documents_directory_exists": os.path.exists("./documents")
    }

def ensure_directories_exist():
    """Ensure all required directories exist"""
    directories = [BASE_PATH,CHROMA_DB_PATH]
    
    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")
            

tools = [
    Tool(
        name="ListServers",
        func=list_servers,
        description=(
            "Use this tool when the user asks to view, list, or summarize all available servers. "
            "Triggers include: 'List all servers', 'Show servers being monitored', 'What servers are active?'. "
            "Returns a human-readable summary of all monitored servers, including:\n"
            "Use this tool to list all monitored servers. "
            "Input should be an empty string. "
            "Example: Action: ListAllServers[]"
            "- Serial number\n"
            "- Last seen timestamp\n"
            "- Total records\n"
            "- Average CPU usage\n"
            "- Average ambient temperature"
        ),
        return_direct=True
    ),
    Tool(
    name="GetTopServersByCPUUtil",
    func=get_top_servers_by_cpu_util,
    description=(
        "Use this tool to retrieve servers with the highest CPU utilization. "
        "Extracts how many top servers to show from the query (default: 10; 'all' returns all). "
        "including numeric words such as 'one server', 'two servers', 'three servers', etc. (default: 10; 'all' returns all). "
        "tell me the top 100 server which have highest cpu utlization\n "
         "tell me the top 100 server which have highest cpu utlization if the count of server is greater tahn list server then return default\n "
        "Example inputs:\n"
        "- 'Top 5 CPU servers'\n"
        "- 'Which 3 servers have highest CPU utilization?'\n"
        "- 'Show all high CPU servers'\n\n"
        "Returns for each server:\n"
        "- Serial number\n"
        "- Peak CPU (%)\n"
        "- Timestamp\n"
        "- Power (Watts)\n"
        "- Temperature (°C)\n"
        "- Fan speed (RPM)\n\n"
        "Format: Action: GetTopServersByCPUUtil[\"<query>\"]"
    ),
    return_direct=True),
    
Tool(
    name="GetServerCPUUtilization",
    func=get_specific_server_cpu_utilization,
    description=(
        "Use this tool to get detailed CPU utilization information for SPECIFIC server(s).\n\n"
        "The tool uses robust server identification to handle various formats including:\n"
        "- Full server names (e.g., 'server SGH227WTNK')\n"
        "- Direct serial numbers (e.g., 'SGH227WTNK')\n"
        "- Multiple servers in one query (e.g., 'SGH227WTNK and ABC123', 'SGH227WTNK, DEF456')\n"
        "- Case-insensitive matching\n"
        "- Handles common typos and formatting variations\n\n"
        "Example queries:\n"
        "- 'What is the CPU utilization of server SGH227WTNK?'\n"
        "- 'Show CPU stats for SGH227WTNK and ABC123'\n"
        "- 'CPU usage for servers SGH227WTNK, DEF456'\n"
        "- 'SGH227WTNK CPU utilization details'\n"
        "- 'Compare CPU usage SGH227WTNK ABC123'\n\n"
        "Returns detailed analysis:\n"
        "For single server:\n"
        "- Average CPU Utilization (%)\n"
        "- Peak CPU Utilization (%) with timestamp\n"
        "- Power consumption at peak\n"
        "- Temperature and fan speed at peak\n"
        "- CPU efficiency rating\n"
        "- Fleet ranking position\n\n"
        "For multiple servers:\n"
        "- Summary statistics across servers\n"
        "- Individual server breakdowns\n"
        "- Comparative analysis with rankings\n\n"
        "If server(s) not found, returns helpful error message with available server examples.\n\n"
        "Format: Action: GetServerCPUUtilization[\"<query>\"]"
    ),
    return_direct=True
),
    
    Tool(
    name="GetLowestServersByCPUUtil",
    func=get_lowest_servers_by_cpu_util,
    description=(
        "Use this tool to find servers with the lowest CPU utilization. It extracts how many servers to show from the user's query, "
        "including numeric words such as 'one server', 'two servers', 'three servers', etc. (default: 10; 'all' returns all). "
         "tell me the top 100 server which have Lowest cpu utlization\n "
         "tell me the top 100 server which have Lowest cpu utlization if the count of server is greater tahn list server then return default\n "
        "Example queries: "
        "'Top 5 servers with lowest CPU utilization', "
        "'Which three servers have the lowest CPU usage?', "
        "'Show all low CPU servers', "
        "'List one server with lowest CPU utilization'. "
        "Returns for each server: "
        "- Serial number "
        "- Lowest CPU (%) "
        "- Timestamp "
        "- Power (Watts) "
        "- Temperature (°C) "
        "- Fan speed (RPM)"
    ),
    return_direct=True
),
Tool(
    name="GetTopServersByAmbientTemp",
    func=get_top_servers_by_ambient_temp,
    description=(
        "Use this tool to find servers ranked by their highest ambient temperature records. "
        "It interprets the user's query to extract how many top servers to show, including numeric words such as "
         "tell me the top 100 server which have highest Ambient Temprature\n "
         "tell me the top 100 server which have highest Ambient Temprature if the count of server is greater tahn list server then return default\n "
        "'one', 'two', 'three', etc., or 'all' for all servers with data. If no number is specified, it shows all servers. "
        "Example queries include: "
        "'Top 5 servers with highest ambient temperature', "
        "'Which three servers have the highest ambient temperature?', "
        "'Show all servers by ambient temperature', "
        "'List one server with highest ambient temperature'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Highest ambient temperature (°C) "
        "- Timestamp of the highest temperature record "
        "- CPU utilization (%) at that time "
        "- CPU power consumption (Watts) "
        "- DIMM power consumption (Watts) "
        "Handles incomplete data gracefully and informs if requested count exceeds available data."
    ),
    return_direct=True
),
Tool(
    name="GetSpecificServerAmbientTemp",
    func=get_specific_server_ambient_temp,
    description=(
        "Use this tool to get ambient temperature data for specific server(s) identified by their serial numbers. "
        "It can handle single or multiple servers in one query with robust server identification. "
        "The function accepts natural language queries containing server serial numbers and returns detailed "
        "ambient temperature information. "
        "Example queries include: "
        "'What is the ambient temperature for server SGH227WTNK?', "
        "'Show ambient temperature data for SGH227WTNK and ABC123', "
        "'Get temperature info for server XYZ456', "
        "'SGH227WTNK, DEF456 ambient temperature'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Highest ambient temperature recorded (°C) "
        "- Average ambient temperature (°C) "
        "- Timestamp of the highest temperature record "
        "- CPU utilization (%) at peak temperature "
        "- CPU power consumption (Watts) at peak "
        "- DIMM power consumption (Watts) at peak "
        "Handles multiple servers with summary statistics and gracefully handles missing data. "
        "Uses the same robust server identification patterns as the CO2 emission function."
    ),
    return_direct=True
),
Tool(
    name="GetLowestServersByAmbientTemp",
    func=get_lowest_servers_by_ambient_temp,
    description=(
        "Use this tool to find servers ranked by their lowest ambient temperature records. "
         "tell me the top 100 server which have lowest ambient temperature\n "
         "tell me the top 100 server which have lowest ambient temperature if the count of server is greater tahn list server then return default\n "
        "It interprets the user's query to extract how many bottom servers to show, including numeric words such as "
        "'one', 'two', 'three', etc., or 'all' for all servers with data. If no number is specified, it shows all servers. "
        "Example queries include: "
        "'Bottom 5 servers with lowest ambient temperature', "
        "'Which three servers have the lowest ambient temperature?', "
        "'Show all servers by lowest ambient temperature', "
        "'List one server with lowest ambient temperature'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Lowest ambient temperature (°C) "
        "- Timestamp of the lowest temperature record "
        "- CPU utilization (%) at that time "
        "- CPU power consumption (Watts) "
        "- DIMM power consumption (Watts) "
        "Handles incomplete data gracefully and informs if requested count exceeds available data."
    ),
    return_direct=True
),
Tool(
    name="GetTopServersByPeak",
    func=get_top_servers_by_peak,
    description=(
        "Use this tool to retrieve servers with the highest peak values across all metrics. "
        "The number of top servers to show is extracted from the user query. "
        "tell me the top 100 server which have highest peak values\n "
         "tell me the top 100 server which have highest peak values if the count of server is greater tahn list server then return default\n "
        "Handles numeric expressions like 'one server', 'top 3 servers', 'all peak servers', etc. "
        "Defaults to 10 if unspecified. 'All' returns the full list.\n\n"
        "Example queries:\n"
        "- 'Which 5 servers have the highest peak values?'\n"
        "- 'Top 3 servers by peak usage'\n"
        "- 'Show all servers with highest peak value'\n\n"
        "For each server, returns:\n"
        "- Serial number\n"
        "- Highest peak value\n"
        "- Timestamp of peak\n"
        "- CPU Utilization (%)\n"
        "- Ambient Temperature (°C)\n"
        "- CPU Power (Watts)\n\n"
        "Format: Action: GetTopServersByPeak[\"<query>\"]"
    ),
    return_direct=True
),
Tool(
    name="GetSpecificServerPeakData",
    func=get_specific_server_peak_data,
    description=(
        "Use this tool to get peak data for specific server(s) identified by their serial numbers. "
        "It can handle single or multiple servers in one query with robust server identification. "
        "The function accepts natural language queries containing server serial numbers and returns detailed "
        "peak performance information. "
        "Example queries include: "
        "'What is the peak data for server SGH227WTNK?', "
        "'Show peak values for SGH227WTNK and ABC123', "
        "'Get peak performance for server XYZ456', "
        "'SGH227WTNK, DEF456 peak data'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Highest peak value recorded "
        "- Average peak value "
        "- Timestamp of the highest peak record "
        "- CPU utilization (%) at peak "
        "- Ambient temperature (°C) at peak "
        "- CPU power consumption (Watts) at peak "
        "Handles multiple servers with summary statistics and gracefully handles missing data. "
        "Uses the same robust server identification patterns as other specific server functions."
    ),
    return_direct=True
),
Tool(
    name="GetLowestServersByPeak",
    func=get_lowest_servers_by_peak,
    description=(
        "Use this tool to retrieve servers with the lowest peak values across all metrics. "
        "The number of servers to show is extracted from the user query. "
         "tell me the top 100 server which have  Lowest peak values\n "
         "tell me the top 100 server which have lowest peak values if the count of server is greater tahn list server then return default\n "
        "Handles phrases like 'one server', 'bottom 3 servers', 'all low peak servers', etc. "
        "Defaults to 10 if not specified. 'All' returns the full list.\n\n"
        "Example queries:\n"
        "- 'Show 3 servers with the lowest peak usage'\n"
        "- 'Bottom 5 peak value servers'\n"
        "- 'All servers with the lowest peak values'\n\n"
        "For each server, returns:\n"
        "- Serial number\n"
        "- Lowest peak value\n"
        "- Timestamp of that value\n"
        "- CPU Utilization (%)\n"
        "- Ambient Temperature (°C)\n"
        "- CPU Power (Watts)\n\n"
        "Format: Action: GetLowestServersByPeak[\"<query>\"]"
    ),
    return_direct=True
),
Tool(
    name="GetServerStats",
    func=get_server_stats,
    description=(
        "Use this tool to retrieve statistics for a specific server or a summary of the fleet.\n\n"
        "Triggers include:\n"
        "- 'Stats for server ABC123'\n"
        "- 'Give me server ST-998 details'\n"
        "- 'Show latest observation for server Y56-22'\n"
        "- 'Show latest observation for all the server '\n"
        "- 'Show latest observation for each  server'\n"
        "- 'What’s the summary of all servers?'\n\n"
        "If a specific server serial number is mentioned in the query, this tool returns:\n"
        "- Latest record timestamp\n"
        "- CPU Utilization, Peak Value, Power (W), Ambient Temperature\n"
        "- Peak and lowest CPU with timestamps\n"
        "- Max/min ambient temperatures with timestamps\n"
        "- Estimated total energy used and CO₂ emissions\n\n"
        "If no server is specified, it returns a fleet-wide summary:\n"
        "- Top 5 servers by peak CPU usage\n"
        "- Latest CPU and temperature readings\n"
        "- General fleet statistics and record availability\n\n"
        "Example inputs:\n"
        "- 'Show stats for server TDX-901'\n"
        "- 'Fleet summary'\n"
        "- 'Give observation for server XP100'\n\n"
        "Format: Action: GetServerStats[\"<query>\"]"
    ),
    return_direct=True
),

# Updated Tool Definitions
Tool(
    name="CalculateCarbonFootprint",
    func=calculate_carbon_footprint,
    description=(
        "Use this tool to calculate the carbon footprint of multiple servers or fleet-wide analysis.\n\n"
        "The tool determines whether to use 'average', 'low-carbon', or 'high-carbon' grid intensity based on keywords in the query "
        "(e.g., 'renewable', 'coal', 'low carbon', etc.). It extracts server counts (like 'top 5', 'ten servers', or 'all') "
        "and optionally filters by grid type.\n\n"
        "This tool is for MULTIPLE server analysis only. For individual servers, use CO2EmissionServer instead.\n\n"
        "Example queries:\n"
        "- 'Show CO2 emissions for all servers using renewable energy.'\n"
        "- 'Calculate carbon footprint for top 3 servers using high carbon grid.'\n"
        "- 'List servers with highest emissions based on coal grid.'\n"
        "- 'Top 10 servers by carbon emissions'\n"
        "- 'Fleet-wide carbon footprint analysis'\n\n"
        "Returns fleet summary including:\n"
        "- Total CO₂ Emissions across all servers\n"
        "- Average emissions per server\n"
        "- Top N highest emitting servers with details\n"
        "- Energy efficiency distribution\n\n"
        "Format: Action: CalculateCarbonFootprint[\"<query>\"]"
    ),
    return_direct=True
),

Tool(
    name="CO2EmissionServer",
    func=co2_emission_server,
    description=(
        "Use this tool to calculate the carbon footprint of SPECIFIC server(s) - single or multiple.\n\n"
        "The tool uses robust server identification to handle various formats including:\n"
        "- Full server names (e.g., 'server SGH227WTNK')\n"
        "- Direct serial numbers (e.g., 'SGH227WTNK')\n"
        "- Multiple servers in one query (e.g., 'SGH227WTNK and ABC123', 'SGH227WTNK, DEF456')\n"
        "- Case-insensitive matching\n"
        "- Handles common typos and formatting variations\n\n"
        "Also determines carbon intensity based on keywords in the query "
        "(e.g., 'renewable', 'coal', 'low carbon', etc.).\n\n"
        "Example queries:\n"
        "- 'What is the carbon footprint of server SGH227WTNK?'\n"
        "- 'CO2 emissions for SGH227WTNK and ABC123 using renewable energy'\n"
        "- 'Show carbon footprint of servers SGH227WTNK, DEF456 with high carbon grid'\n"
        "- 'SGH227WTNK ABC123 carbon emissions'\n"
        "- 'Compare CO2 for SGH227WTNK and DEF456'\n\n"
        "Returns detailed analysis:\n"
        "For single server:\n"
        "- Energy Consumed (kWh)\n"
        "- CO₂ Emissions (kg)\n"
        "- Carbon Intensity used in calculation\n"
        "- Average CPU Utilization (%)\n"
        "- Energy Efficiency Rating\n\n"
        "For multiple servers:\n"
        "- Total and average CO₂ emissions\n"
        "- Individual server breakdowns\n"
        "- Comparative analysis\n\n"
        "If server(s) not found, returns helpful error message with available server examples.\n\n"
        "Format: Action: CO2EmissionServer[\"<query>\"]"
    ),
    return_direct=True
),

Tool(
    name="CalculateCarbonFootprintLowest",
    func=calculate_carbon_footprint_lowest,
    description=(
        "Use this tool to calculate the carbon footprint of one or more servers based on estimated energy consumption, "
        "specifically showing servers with the LOWEST CO₂ emissions.\n\n"
        "The tool determines whether to use 'average', 'low-carbon', or 'high-carbon' grid intensity based on keywords in the query "
        "(e.g., 'renewable', 'coal', 'low carbon', etc.). It extracts server serial numbers, counts (like 'top 5', 'ten servers', or 'all'), "
        "and optionally filters by grid type.\n\n"
        "If a specific server is mentioned (e.g., 'Server ABC123'), it will return the carbon footprint details for that server only.\n"
        "If a number of top servers are requested, it returns those with the LOWEST CO₂ emissions (most energy-efficient).\n\n"
        "Example queries:\n"
        "- 'Show me the 10 most energy-efficient servers'\n"
        "- 'Which servers have the lowest carbon footprint?'\n"
        "- 'Top 5 cleanest servers using renewable energy'\n"
        "- 'List servers with minimum CO2 emissions'\n"
        "- 'Show least polluting servers'\n"
        "- 'Most efficient servers by carbon footprint'\n\n"
        "Returns for each server:\n"
        "- Serial number\n"
        "- Energy Consumed (kWh)\n"
        "- CO₂ Emissions (kg)\n"
        "- Average CPU Utilization (%)\n"
        "- Efficiency Rating (based on CPU-to-power ratio)\n\n"
        "Also returns a fleet summary when multiple servers are included, with efficiency distribution.\n\n"
        "Format: Action: CalculateCarbonFootprintLowest[\"<query>\"]"
    ),
    return_direct=True
),

Tool(
    name="IdentifyHighCPUServers",
    func=identify_high_cpu_servers,
    description=(
        "Use this tool to identify servers that have CPU utilization above a specified threshold. "
        "The query should include a numeric threshold (e.g., 'above 80%' or 'more than 70%'). "
        "This tool analyzes all server records and returns those with at least one instance of CPU utilization above the given threshold.\n\n"
        "Example inputs:\n"
        "- 'Show servers with CPU above 90%'\n"
        "- 'List all servers crossing 75% CPU utilization'\n"
        "- 'Which servers hit CPU over 85%?'\n\n"
        "Returns for each matching server:\n"
        "- Serial number\n"
        "- Count and percentage of records where CPU > threshold\n"
        "- Maximum CPU utilization observed\n\n"
        "Notes:\n"
        "- Maximum 20 servers are shown in detail; remaining are summarized.\n"
        "- Results are sorted by percentage of high CPU records and peak CPU observed.\n\n"
        "Format: Action: IdentifyHighCPUServers[\"<query>\"]"
    ),
    return_direct=True,
),
Tool(
    name="GetServerTimestamps",
    func=get_server_timestamps,
    description=(
        "Use this tool to retrieve the list of timestamped monitoring records for a specific server. "
        "The query must contain the server serial number (e.g., 'server A123B', 'timestamps for server XYZ001'). "
        "This helps in auditing the monitoring intervals or understanding activity history.\n\n"
        "Example inputs:\n"
        "- 'Show timestamps for server A12B9'\n"
        "- 'Get monitoring history of server 7GHT9'\n"
        "- 'When was server TEST-SRVR last active?'\n\n"
        "Returns:\n"
        "- The server’s total number of records\n"
        "- Up to 20 of the earliest timestamps (in order of appearance in data)\n"
        "- A note about any additional timestamps\n\n"
        "Notes:\n"
        "- The tool tries to match server serials even with partial or imprecise queries.\n"
        "- If no match is found, it will suggest a few available server serials.\n\n"
        "Format: Action: GetServerTimestamps[\"<query>\"]"
    ),
    return_direct=True,
),
Tool(
    name="FilterServerRecords",
    func=get_filtered_server_records,
    description=(
        "Use this tool to filter and retrieve monitoring records for a specific server based on a metric condition. "
        "The input must be a JSON string specifying the server serial, metric, comparison operator, and value.\n\n"
        "Supported metrics:\n"
        "- 'cpu_util': CPU utilization (%)\n"
        "- 'amb_temp': Ambient temperature (°C)\n"
        "- 'peak': Peak performance value\n\n"
        "Supported operators:\n"
        "- 'greater_than': Metric is greater than the given value\n"
        "- 'less_than': Metric is less than the given value\n"
        "- 'equals': Metric is equal to the given value\n\n"
        "Example inputs:\n"
        "- '{\"server_serial\": \"SRV123\", \"metric\": \"cpu_util\", \"operator\": \"greater_than\", \"value\": 80}'\n"
        "- '{\"server_serial\": \"ABC456\", \"metric\": \"amb_temp\", \"operator\": \"less_than\", \"value\": 25}'\n\n"
        "Returns:\n"
        "- A list of up to 20 matching records (timestamp and metric value)\n"
        "- Total count of matching entries\n"
        "- A message if no records matched\n\n"
        "Notes:\n"
        "- Ensure all fields are correctly specified in double-quoted JSON.\n"
        "- Server serial is case-insensitive.\n"
        "- Returns an error message if fields are missing or invalid.\n\n"
        "Format: Action: FilterServerRecords[\"<JSON-formatted string>\"]"
    ),
    return_direct=True,
),
Tool(
    name="DetectAnomalies",
    func=detect_anomalies,
    description=(
        "Use this tool to analyze server monitoring data and detect significant anomalies across key performance metrics. "
        "The tool uses statistical analysis (modified Z-score with median and MAD) to identify abnormal spikes in CPU utilization, temperature, and power usage.\n\n"
        
        "You can run this tool for all servers or specify a particular server or metric in natural language.\n\n"
        
        "Supported metrics (automatically inferred from query):\n"
        "- 'cpu_util': CPU utilization (%)\n"
        "- 'amb_temp': Ambient temperature (°C)\n"
        "- 'cpu_watts': CPU power consumption (watts)\n"
        "- 'dimm_watts': DIMM memory power usage (watts)\n\n"
        
        "Example queries:\n"
        "- 'Check anomalies for server SRV123'\n"
        "- 'Analyze CPU utilization across all servers'\n"
        "- 'Find temperature spikes in ABC789'\n"
        "- 'Show anomalies in power consumption'\n\n"
        
        "Returns:\n"
        "- Enhanced anomaly report (text) including:\n"
        "  • Median baseline values for each metric\n"
        "  • Critical anomalies (Z-score > 5)\n"
        "  • Major anomalies (3.5 < Z-score ≤ 5)\n"
        "  • Frequent times of anomalies (hour-level)\n"
        "  • Suggested root causes based on patterns\n\n"
        
        "Notes:\n"
        "- The function works with historical records stored for each server.\n"
        "- If no significant anomalies are detected, a clean report is returned.\n"
        "- Supports natural language input only.\n\n"
        
        "Format: Action: DetectAnomalies[\"<natural language query>\"]"
    ),
    return_direct=True,
),
 Tool(
        name="QueryDocuments",
        func=query_documents,
        description=(
            "Use this tool to search and query the HPE Energy Efficiency and Sustainability knowledge base. "
            "This tool provides comprehensive guidance on HPE server energy optimization, carbon emission reduction, "
            "and sustainable data center operations. "
            "Triggers include: 'How to reduce server power consumption', 'HPE energy efficiency recommendations', "
            "'PUE optimization strategies', 'Carbon footprint reduction', 'HPE iLO power management', "
            "'Thermal efficiency issues', 'Server consolidation advice', 'HPE OneView energy features'. "
            "Input should be a specific question about:\n"
            "- HPE server energy efficiency (Power Usage Effectiveness, Energy Efficiency Rating)\n"
            "- HPE-specific power management technologies (iLO, Dynamic Power Capping, Power Regulator)\n"
            "- Carbon Usage Effectiveness (CUE) and emissions tracking\n"
            "- Renewable energy integration strategies\n"
            "- Thermal management and cooling optimization\n"
            "- HPE server hardware efficiency recommendations (Gen10/Gen11 ProLiant)\n"
            "- Data center sustainability compliance (ASHRAE 90.4, ISO 14001/50001, Energy Star)\n"
            "- HPE infrastructure optimization (OneView, InfoSight, Synergy)\n"
            "Example: 'How can I optimize power consumption on HPE ProLiant Gen11 servers with high idle power?'"
        ),
        return_direct=True
    ),
    
    Tool(
        name="ListAvailableDocuments",
        func=list_available_documents,
        description=(
            "Use this tool to list all available HPE energy efficiency and sustainability documents in the knowledge base. "
            "This tool helps users understand what specific HPE energy optimization resources are available for querying. "
            "Triggers include: 'What HPE energy documents do you have?', 'List available sustainability guides', "
            "'Show HPE efficiency documentation', 'What energy resources are loaded?', 'Available HPE knowledge base files'. "
            "Returns information about:\n"
            "- HPE Energy Efficiency Standards and Guidelines\n"
            "- HPE Server Power Management Documentation\n"
            "- Carbon Emission Reduction Frameworks\n"
            "- HPE Data Center Sustainability Best Practices\n"
            "- HPE-Specific Tool Integration Guides (iLO, OneView, InfoSight)\n"
            "- Compliance and Certification Documentation\n"
            "- HPE Hardware Optimization Manuals\n"
            "Input should be an empty string or 'list'. "
            "Example usage: Action: ListAvailableDocuments[]"
        ),
        return_direct=True
    )
]
# -----------------------------
# Agent Setup
# -----------------------------
# Agent Setup
# -----------------------------
def setup_agent(current_llm):
    global llm_instance # Ensure globals are accessible if needed by tools
    # llm_instance is already set globally in IntegratedChatSystem.__init__
# In the setup_agent function:

    # ... (system_template up to Key Reminders) ...
    system_template = """You are a Server Monitoring Assistant.
    if any type of greeting query come then you respond it not use the tools
Your primary task is to respond to user queries by selecting the appropriate tool and providing the correct input for it.
You MUST ALWAYS use the ReAct format when a tool is needed.

**IMPORTANT: All tools return their output as formatted strings that are ready to display to the user. Do NOT attempt to parse, modify, or reformat tool outputs - they are already properly formatted for presentation.**

**MANDATORY ReAct FORMAT (Thought, Action, Action Input):**
When you need to use a tool, you MUST output your reasoning and the tool call in this EXACT, multi-line format. Each part MUST be on its own line.

Thought: [Your concise reasoning for choosing the tool and formulating the input. Always include this line.]
Action: [The EXACT name of ONE tool from the available tools list. Always include this line. Do NOT invent tool names.]
Action Input: [The specific input string or JSON for the chosen tool. Always include this line. If the tool description says input is an empty string or not strictly needed, provide an empty string or a relevant placeholder like "all" or the query itself if appropriate. For JSON, provide a single-line, correctly formatted JSON string with double quotes for keys and string values.]

After outputting Action Input, STOP. The system will provide an Observation with the tool's string output.
DO NOT add any other text, commentary, or formatting around or before the Thought, Action, or Action Input lines.
DO NOT write "Observation:" yourself.
DO NOT attempt to process or reformat the tool's output - it comes pre-formatted as a string ready for display.

Example 1 (Tool expecting simple text, possibly empty):
Thought: The user wants to list all servers. I should use the ListServers tool. The tool description says Action Input can be an empty string.
Action: ListServers
Action Input:

Example 2 (Tool expecting specific text):
Thought: The user wants the top 3 CPU servers. I should use the TopCPUUtilizationServers tool and specify 'top 3'.
Action: TopCPUUtilizationServers
Action Input: top 3

Example 3 (Tool expecting JSON):
Thought: The user wants records for server SGH123 where CPU utilization is greater than 50. I should use GetFilteredServerRecords with the specified JSON structure.
Action: GetFilteredServerRecords
Action Input: {{"server_serial": "SGH123", "metric": "cpu_util", "operator": "greater_than", "value": 50}}

If the question is a simple greeting, a follow-up to a previous tool's direct output (where return_direct=true), or does not require a tool (e.g., "hello", "thank you"), respond directly using the Final Answer format:
Thought: The user said hello. I should respond in kind.
Final Answer: Hello! How can I help you today?

**Available Tools Information:**
{tools}

**Tool Names for the 'Action:' line (Case Sensitive):**
{tool_names}

**Key Reminders:**
- Your response, when using a tool, MUST strictly follow the Thought, Action, Action Input format. Each part MUST be on its own line.
- The 'Action:' line MUST contain ONLY the exact tool name from the 'Tool Names for the Action line' list. Do NOT add any other characters, markdown (like '**' or '_'), quotes, or extra newlines to the tool name itself on this line. It must be plain text.
- The 'Action Input:' line MUST follow immediately after the 'Action:' line.
- For `GetFilteredServerRecords`, the Action Input MUST be the JSON string. If the user says "server X cpu > 10", you must translate that to the JSON.
- Most tools have `return_direct=True`. Their output goes straight to the user. Do not re-process it unless asked a follow-up.
- All tool outputs are already formatted as complete, readable strings - DO NOT attempt to parse, modify, or add formatting to them.
- If you are providing a final answer directly without a tool, use the 'Final Answer:' format. Do NOT mix 'Final Answer:' with 'Action:' blocks in the same response. Once you output 'Final Answer:', your response for that turn is complete.
- Do NOT use any ANSI escape codes (like color codes such as `[0m`) in your Thought, Action, Action Input, or Final Answer sections. All these sections must be plain text.

Conversation History (for context):
{chat_history}

User's current question:
{input}

Agent scratchpad (your previous thoughts, actions, and observations for this current question):
{agent_scratchpad}"""

    # ... (rest of the setup_agent function)

    # Using ChatPromptTemplate.from_template to ensure proper handling by create_react_agent
    prompt = ChatPromptTemplate.from_template(system_template)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        max_token_limit=1500 # Increased slightly
    )

    def enhanced_parsing_error_handler(error: Any) -> str:
        error_str = str(error)
        logger.error(f"Agent parsing/execution error: {error_str}", exc_info=True) # Full traceback for server logs

        # Try to extract a Final Answer if the LLM put it in the error message
        final_answer_match = re.search(r"Final Answer:\s*(.*)", error_str, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            extracted_answer = final_answer_match.group(1).strip()
            if len(extracted_answer) > 5: # Basic check for non-empty answer
                logger.info(f"Extracted a potential Final Answer from error: {extracted_answer[:100]}...")
                return extracted_answer

        # Default guidance message
        guidance = " To help me process your request, please try:\n" \
                   "- Phrasing your question more simply or specifically.\n" \
                   "- Asking for less data (e.g., 'top 3 servers' instead of 'all').\n" \
                   "- Breaking complex questions into smaller parts."

        if "Could not parse LLM output" in error_str or "Could not parse tool invocation" in error_str:
             return "I had a little difficulty formatting my response or deciding the next step. Could you please rephrase your question?" + guidance
        if "Invalid tool" in error_str:
            return "I seem to have chosen an incorrect tool. Please rephrase your request, and I'll try a better one." + guidance
        if "no viable tool" in error_str.lower() or "did not find an Action" in error_str: # Agent couldn't decide on a tool
            return "I couldn't determine the best way to handle that request with my current tools. Could you try asking differently?" + guidance

        # Generic fallback for other errors caught by this handler
        return "I'm sorry, an unexpected issue occurred while I was thinking." + guidance


    try:
        # First create the agent using create_react_agent
        agent = create_react_agent(llm=current_llm, tools=tools, prompt=prompt)
        
        # Then create the AgentExecutor with the agent
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=enhanced_parsing_error_handler, # Use the robust handler
            max_iterations=3, # Allow a bit more room for complex interactions or retries
            early_stopping_method="force",
            max_execution_time=100, # Slightly longer timeout
            return_intermediate_steps=False, # Keep False for production, True for deep debugging
        )
        logger.info("AgentExecutor created successfully with create_react_agent and refined prompt.")
        return agent_executor

    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Agent creation failed: {e}", exc_info=True)
        # This will be caught by create_gradio_interface's error handling
        raise RuntimeError(f"Agent setup failed, application cannot continue: {e}") from e


# -----------------------------
# Enhanced Chat System
# -----------------------------
class IntegratedChatSystem:
    def __init__(self):
        global llm_instance # Declare intent to modify/set globals
        self.logger = logging.getLogger(self.__class__.__name__)
        self.semaphore = threading.Semaphore(1)

        self.logger.info("Initializing IntegratedChatSystem...")
        # Ensure llm_instance is assigned to the global
        llm_instance = LLM_INSTANCE

        self.logger.info(f"ChatOllama model '{llm_instance.model}' initialized and assigned globally.")

         # Auto-initialize RAG system
        rag_success = auto_initialize_rag_on_startup(llm_instance)
        if rag_success:
            self.logger.info("RAG system auto-initialization completed successfully")
        else:
            self.logger.warning("RAG system auto-initialization failed, but continuing with setup")

        self.agent_executor = setup_agent(llm_instance) # Pass the global instance
        
        # Log final RAG status
        rag_status = get_rag_system_status()
        self.logger.info(f"Final RAG system status: {rag_status}")

        if self.agent_executor:
            self.logger.info("Chat agent setup complete.")
        else:
            # setup_agent now raises RuntimeError on failure, so this else might not be hit
            # unless setup_agent is changed to return None on some non-critical error.
            self.logger.critical("Chat agent setup did not return an executor. System may not function.")
            # This would typically be caught by the RuntimeError in setup_agent
            # raise RuntimeError("Agent Executor could not be initialized.")


    def clean_response_text(self, response_text: str) -> str:
        # Remove common error prefixes that might slip through
        prefixes_to_remove = [
            r'^Could not parse LLM output:.*?For troubleshooting.*?\n',
            r'^I encountered a parsing error:.*?\n',
            r'^Format Answering Task Error:.*?\n',
            # Add more specific prefixes if observed
        ]
        for prefix in prefixes_to_remove:
            response_text = re.sub(prefix, '', response_text, flags=re.DOTALL | re.IGNORECASE)

        # Prioritize extracting "Final Answer:" if present, as per ReAct convention
        final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            response_text = final_answer_match.group(1).strip()
        else:
            # If no "Final Answer:", check if it's an Observation (tool output)
            # This part is tricky because `return_direct=True` tools bypass this.
            # This cleanup is more for when `return_direct=False` or if the agent is trying to summarize.
            # For now, this simpler cleanup should be okay since most tools are direct.
            pass

        # Remove any remaining ReAct keywords if the LLM didn't stop cleanly after a Final Answer
        # This is a safeguard.
        react_keywords_pattern = r"\n*\s*(Thought:|Action:|Action Input:|Observation:).*"
        # Split by the first occurrence of any ReAct keyword block
        parts = re.split(react_keywords_pattern, response_text, 1)
        response_text = parts[0].strip()


        response_text = re.sub(r'\n{3,}', '\n\n', response_text) # Normalize multiple newlines
        return response_text.strip()

    def process_query(self, query: str, chat_history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]: # Replaced typing.List/Tuple
        if not self.semaphore.acquire(timeout=0.5):
            self.logger.warning("Semaphore acquisition timeout. Request concurrent or system busy.")
            # Optionally return a message to display in chat history:
            # chat_history.append((query, "The assistant is busy. Please try again in a moment."))
            # return "", chat_history # This would clear the input box
            return query, chat_history # Keep query in input box for user to retry

        try:
            self.logger.info(f"Processing query: '{query}'")

            if not query.strip():
                response = "Please enter a question."
                chat_history.append((query, response))
                return "", chat_history # Clear input box, update history

            if not self.agent_executor:
                # This should ideally be caught at startup by create_gradio_interface
                self.logger.error("Agent executor is not initialized!")
                response = "The assistant is not properly initialized. Please check system logs."
                chat_history.append((query, response))
                return "", chat_history

            # The agent's memory (ConversationBufferWindowMemory) handles chat_history.
            # We pass it as `chat_history` in the input dict if the prompt expects it.
            # The ReAct prompt template `system_template` has `{chat_history}`.
            # The memory object formats this history appropriately.
            agent_input = {"input": query} # `chat_history` is implicitly handled by the memory attached to AgentExecutor

            agent_response_dict = self.agent_executor.invoke(agent_input)

            # `output` is the key AgentExecutor uses for the final response or tool output if direct.
            response_text = agent_response_dict.get("output", "Sorry, I could not process that effectively right now.")

            cleaned_response = self.clean_response_text(response_text)

            if not cleaned_response or len(cleaned_response) < 3: # Stricter check for emptiness
                 # This might happen if a tool returns an empty string and `return_direct=True`
                 # Or if the LLM gives a very terse "Final Answer:"
                 cleaned_response = "I processed your request, but there wasn't much to say, or the result was empty. Can I help with something else?"
                 self.logger.warning(f"Agent returned a very short/empty response. Original: '{response_text}', Using: '{cleaned_response}'")

            chat_history.append((query, cleaned_response))
            self.logger.info(f"Successfully processed query. Response: '{cleaned_response[:100]}...'")
            return "", chat_history # Clear input box

        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}", exc_info=True)
            # Provide a user-friendly error message
            error_response_msg = "I encountered a system error while processing your request. Please try rephrasing or simplify your query. If the problem continues, please check the logs or contact support."
            chat_history.append((query, error_response_msg))
            return "", chat_history # Clear input box
        finally:
            self.semaphore.release()


# -----------------------------
# Gradio UI
# -----------------------------
def create_gradio_interface():
    logger.info("Creating Gradio interface...")
    try:
        # This will initialize llm_instance globally
        chat_system = IntegratedChatSystem()
        logger.info("IntegratedChatSystem initialized for Gradio.")

    except RuntimeError as e: # Catch specific RuntimeError from agent/system setup
        logger.critical(f"Fatal error during Gradio system initialization (RuntimeError): {e}", exc_info=True)
        with gr.Blocks(title="Error - Assistant Unavailable", theme=gr.themes.Soft()) as demo_error:
            gr.Markdown("## 🖥️ Assistant Initialization Failed")
            gr.Markdown(f"A critical runtime error occurred during startup: **{str(e)}**")
            gr.Markdown("The assistant is currently unavailable. Please check the application logs for more details. Ensure all dependencies (like Ollama server) are running correctly.")
        return demo_error
    except Exception as e: # Catch any other unexpected errors during setup
        logger.critical(f"Unexpected fatal error during Gradio system initialization: {e}", exc_info=True)
        with gr.Blocks(title="Error - Assistant Unavailable", theme=gr.themes.Soft()) as demo_error:
            gr.Markdown("## 🖥️ Assistant Initialization Failed")
            gr.Markdown(f"An unexpected error occurred during startup: **{str(e)}**")
            gr.Markdown("The assistant is currently unavailable. Please check the application logs.")
        return demo_error

    # If chat_system initialized successfully, proceed to create the main UI
    with gr.Blocks(title="Server Monitoring Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🖥️ Server Monitoring Assistant")
        gr.Markdown("""
        **Ask about:** Server performance (CPU, temperature, peak), Carbon footprint analysis.
        **Tips:** Be specific (e.g., "Top 3 servers with lowest CPU").
        For filtering, the agent needs to create JSON: "Show records for server SGH123 where CPU is above 50"
        """)

        chatbot_ui = gr.Chatbot(label="Assistant Response", height=600, elem_id="chatbot", show_copy_button=True, bubble_full_width=False)
        msg_input = gr.Textbox(
            label="Your question:",
            placeholder="E.g., 'List all servers', 'Top 3 servers by highest peak CPU', 'Carbon footprint for SGH949WW81 low carbon'",
            lines=2,
            show_label=False # Label is part of the placeholder effectively
        )

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary", scale=3) # Give submit more width
            clear_btn = gr.ClearButton([msg_input, chatbot_ui], value="Clear Chat", scale=1)

        gr.Examples(
            examples=[
                "List all servers",
                "Top 3 servers by highest peak CPU",
                "Carbon footprint for all servers using average grid",
                "Show me records for server SGH949WW81 where amb_temp is greater_than 28", # Agent must translate this to JSON
                "Identify servers with CPU utilization consistently above 85%",
                "What are the timestamps for server SGH001TEST?",
                "Hello there"
            ],
            inputs=msg_input,
            label="Example Questions"
        )

        # Wire up the chat processing function
        submit_btn.click(chat_system.process_query, [msg_input, chatbot_ui], [msg_input, chatbot_ui])
        msg_input.submit(chat_system.process_query, [msg_input, chatbot_ui], [msg_input, chatbot_ui])

    logger.info("Gradio interface created successfully.")
    return demo

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting Server Monitoring Assistant Application...")
    
    if not os.path.exists(BASE_PATH):
        logger.warning("'new_dump1.json' not found. Server data features will be based on empty data.")
    
    # Load server data with error handling
    try:
        with open(Data_PATH, "r") as file:
            server_data_raw = json.load(file)
            logger.info(f"Loaded data for {len(server_data_raw)} servers")
    except FileNotFoundError:
        logger.error("Server data file 'new_dump1.json' not found. Please ensure it's in the same directory.")
        server_data_raw = []
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in server data file 'new_dump1.json'.")
        raise Exception("Invalid JSON format in server data file.")
    
    # Create and launch Gradio interface
    gradio_app = create_gradio_interface()
    try:
        gradio_app.launch(share=False, debug=False)
        logger.info("Application launched successfully. Access via the URL printed above.")
    except Exception as e:
        logger.critical(f"Failed to launch Gradio application: {e}", exc_info=True)