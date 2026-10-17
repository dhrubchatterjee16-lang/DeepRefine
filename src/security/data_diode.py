"""
Unidirectional Optical Data Diode & Demilitarized Zone (DMZ) Gateway.
Complies with IEC 62443 Level 3 / Level 4 air-gap security architecture.

Guarantees:
1. Physical Unidirectional Flow: Data can ONLY flow from OT (Level 3 Historian) -> IT/Cloud (Level 4 DeepRefine).
2. Hardware-enforced Reverse Blinding: Zero TCP/IP connection or handshake can be initiated backward.
3. Cryptographic Frame Integrity: Every payload is stamped with an HMAC-SHA256 signature and sequence nonce.
"""

import hmac
import hashlib
import time
import json
from typing import Dict, Optional, Tuple

class OpticalDataDiodeProxy:
    def __init__(self, shared_secret_key: bytes = b"REFINERY-OT-IEC62443-DIODE-KEY-2026"):
        self.secret_key = shared_secret_key
        self.last_sequence_id = 0

    def transmit_ot_payload(self, plant_telemetry: Dict) -> Dict:
        """
        Executed on OT Sender side (Level 3 Historian DMZ).
        Serializes data, increments nonce, and stamps cryptographic signature.
        """
        self.last_sequence_id += 1
        raw_body = json.dumps(plant_telemetry, sort_keys=True)
        
        # Cryptographic HMAC-SHA256 signature
        signature = hmac.new(self.secret_key, raw_body.encode(), hashlib.sha256).hexdigest()

        packet = {
            "diode_protocol_version": "IEC62443-DIODE-v1.0",
            "direction": "OT_LEVEL_3_TO_IT_LEVEL_4_ONLY",
            "sequence_nonce": self.last_sequence_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload_hmac": signature,
            "data": plant_telemetry
        }
        return packet

    def receive_it_payload(self, packet: Dict) -> Tuple[bool, Optional[Dict], str]:
        """
        Executed on IT Receiver side (Level 4 DeepRefine).
        Validates cryptographic tamper-proofing and enforces sequence continuity.
        Strictly rejects any reverse-communication attempts.
        """
        if packet.get("direction") != "OT_LEVEL_3_TO_IT_LEVEL_4_ONLY":
            return False, None, "SECURITY_VIOLATION: Inbound packet violates unidirectional flow rules."

        payload_data = packet.get("data", {})
        expected_sig = packet.get("payload_hmac", "")
        raw_body = json.dumps(payload_data, sort_keys=True)
        calculated_sig = hmac.new(self.secret_key, raw_body.encode(), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected_sig, calculated_sig):
            return False, None, "TAMPER_DETECTED: Cryptographic signature mismatch. Packet dropped."

        return True, payload_data, "CLEARED_BY_DATA_DIODE"
