"""
Local Industrial SCADA / PLC Emulator Server.
Simulates a live industrial process controller emitting real-time refinery telemetry tags
via a TCP/IP streaming socket on localhost:4840.

Provides realistic fluctuating process dynamics:
- CDU_TopTemp, CDU_BotTemp, CDU_Pressure
- Comp_VibrationX, Comp_VibrationY, Comp_Temp
- Pump_Flow, Pump_Vibration
"""

import socket
import threading
import time
import json
import numpy as np

def run_scada_plc_emulator(host: str = "127.0.0.1", port: int = 4840):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    print(f"[*] Industrial SCADA / PLC Emulator listening on {host}:{port}")

    t = 0
    while True:
        try:
            client, addr = server.accept()
            print(f"[+] DCS Client connected from {addr}")

            while True:
                t += 1
                # Dynamic fluctuating industrial tags
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "CDC_TopTemp": round(115.0 + 1.2 * np.sin(t * 0.1) + np.random.normal(0, 0.2), 2),
                    "CDC_BotTemp": round(345.0 + 2.5 * np.cos(t * 0.05) + np.random.normal(0, 0.4), 2),
                    "CDC_Pressure": round(1.45 + np.random.normal(0, 0.02), 3),
                    "Comp_VibrationX": round(2.1 + 0.15 * np.sin(t * 0.2) + np.random.normal(0, 0.05), 3),
                    "Comp_VibrationY": round(2.0 + 0.15 * np.cos(t * 0.2) + np.random.normal(0, 0.05), 3),
                    "Comp_Temp": round(82.0 + 0.5 * np.sin(t * 0.05) + np.random.normal(0, 0.2), 2),
                    "Pump_Flow": round(420.0 + np.random.normal(0, 3.0), 1),
                    "Pump_Vibration": round(1.8 + np.random.normal(0, 0.08), 3)
                }

                data_bytes = (json.dumps(payload) + "\n").encode("utf-8")
                client.sendall(data_bytes)
                time.sleep(0.5) # 2 Hz industrial polling cadence
        except Exception as e:
            time.sleep(1)

if __name__ == "__main__":
    run_scada_plc_emulator()
