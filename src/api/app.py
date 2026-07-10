"""
Smart Return Fraud Detector API - Minimal Version
Pure Python HTTP server
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
import os

class FraudAPIHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for fraud detection API"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "message": "Smart Return Fraud Detector API",
                "status": "running",
                "version": "1.0.0",
                "endpoints": {
                    "/": "GET - API information",
                    "/health": "GET - Health check",
                    "/predict": "POST - Predict fraud risk"
                }
            }
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "service": "fraud-detector-api",
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Not Found"}
            self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
                
                # Rule-based prediction
                risk_score = 0.0
                features = []
                
                # Return rate
                total_returns = data.get('total_returns', 0)
                total_orders = data.get('total_orders', 1)
                return_rate = total_returns / max(total_orders, 1)
                
                if return_rate > 0.6:
                    risk_score += 0.35
                    features.append({"feature": "High return rate", "value": return_rate, "impact": 0.35})
                
                # Price
                price = data.get('price', 0)
                if price > 200:
                    risk_score += 0.2
                    features.append({"feature": "High value item", "value": price, "impact": 0.2})
                
                # Suspicious reason
                reason = data.get('return_reason', '')
                suspicious = ['Did not like it', 'No longer needed', 'Found better price', 'Changed mind']
                if reason in suspicious:
                    risk_score += 0.2
                    features.append({"feature": "Suspicious reason", "value": reason, "impact": 0.2})
                
                # Near deadline
                days = data.get('days_since_delivery', 0)
                if days > 25:
                    risk_score += 0.15
                    features.append({"feature": "Near deadline", "value": days, "impact": 0.15})
                
                risk_score = min(risk_score, 1.0)
                
                if risk_score >= 0.7:
                    risk_level = "High"
                    prediction = 1
                elif risk_score >= 0.3:
                    risk_level = "Medium"
                    prediction = 1
                else:
                    risk_level = "Low"
                    prediction = 0
                
                response = {
                    "risk_score": round(risk_score, 4),
                    "prediction": prediction,
                    "risk_level": risk_level,
                    "model_used": "rule-based-v1",
                    "timestamp": datetime.now().isoformat(),
                    "top_features": features[:5]
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"error": str(e)}
                self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"error": "Not Found"}
            self.wfile.write(json.dumps(response).encode())

def run_server():
    """Run the HTTP server"""
    port = int(os.getenv('PORT', 8000))
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, FraudAPIHandler)
    print(f"🚀 Server running on port {port}")
    print(f"📍 http://0.0.0.0:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
