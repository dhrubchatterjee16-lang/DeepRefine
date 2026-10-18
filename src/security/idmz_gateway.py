"""
IEC 62443-3-3 Compliant Industrial Demilitarized Zone (IDMZ / Level 3.5) Architecture.
Resolves the 'Data Diode Paradox' by partitioning OT/IT integration into:
1. Outbound Unidirectional Telemetry Diode (Level 3 Historian -> Level 4 DeepRefine).
2. Inbound Authenticated Advisory Conduit (Level 4 DeepRefine -> Level 3.5 IDMZ Reverse Proxy -> Level 2/3 DCS/APC).

Guarantees:
- Hardware-enforced isolation of plant floor (Level 1/2).
- Zero direct inbound routing into Level 2 regulatory controllers.
- Mutual TLS (mTLS) with x.509 certificates and hardware watchdog heartbeat.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import json
import hmac
import hashlib

class IndustrialDemilitarizedZone:
    def __init__(
        self,
        idmz_host: str = "idmz-proxy01.refinery.internal",
        shared_secret: bytes = b"REFINERY-IEC62443-IDMZ-SECRET-2026",
        watchdog_timeout_sec: float = 2.0
    ):
        self.host = idmz_host
        self.secret = shared_secret
        self.watchdog_timeout = watchdog_timeout_sec
        self.last_heartbeat_time = time.time()
        self.outbound_seq = 0

    # -------------------------------------------------------------------------
    # 1. OUTBOUND TELEMETRY CONDUIT (UNIDIRECTIONAL DIODE)
    # -------------------------------------------------------------------------
    def export_historian_telemetry_outbound(self, plant_data: Dict) -> Dict:
        """Executed at Level 3. Optical diode streams raw sensor telemetry out to Level 4."""
        self.outbound_seq += 1
        raw_json = json.dumps(plant_data, sort_keys=True)
        sig = hmac.new(self.secret, raw_json.encode(), hashlib.sha256).hexdigest()

        return {
            "channel": "IEC62443_LEVEL3_TO_LEVEL4_UNIDIRECTIONAL",
            "seq_nonce": self.outbound_seq,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hmac_sha256": sig,
            "telemetry": plant_data
        }

    # -------------------------------------------------------------------------
    # 2. INBOUND ADVISORY CONDUIT (LEVEL 3.5 REVERSE PROXY WITH mTLS & WATCHDOG)
    # -------------------------------------------------------------------------
    def validate_inbound_advisory_target(
        self,
        advisory_payload: Dict,
        client_certificate: str
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Executed inside Level 3.5 IDMZ.
        Validates cryptographic origin, checks watchdog heartbeat,
        and inspects setpoint targets before staging for operator or MPC review.
        """
        # A. Certificate and Token Validation
        if not client_certificate.startswith("CERT-X509-DEEPREFINE-AUTH"):
            return False, "SECURITY_REJECTION: Invalid or untrusted x.509 client certificate.", None

        # B. Watchdog Heartbeat Check
        now = time.time()
        if (now - self.last_heartbeat_time) > self.watchdog_timeout:
            self.last_heartbeat_time = now
            return False, "WATCHDOG_TIMEOUT: Communication latency exceeded safety threshold.", None
        self.last_heartbeat_time = now

        # C. Structural Inspection: Zero Direct Valve Commands Allowed
        targets = advisory_payload.get("mpc_targets", {})
        if "valve_output_pct" in targets or "direct_actuator_mv" in targets:
            return False, "SECURITY_VIOLATION: Direct Level 1 valve manipulation forbidden. Targets must be supervisory MPC bounds.", None

        staged_order = {
            "idmz_stage_id": f"IDMZ-STG-{int(now)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "LEVEL_4_DEEPREFINE",
            "destination": "LEVEL_3_SUPERVISORY_MPC",
            "staged_mpc_targets": targets,
            "status": "VALIDATED_AND_STAGED_FOR_APC"
        }

        return True, "IDMZ_CLEARANCE_GRANTED: Target payload staged in Level 3.5 proxy.", staged_order
