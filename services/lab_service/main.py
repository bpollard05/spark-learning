# services/lab_service/main.py
"""
Lab Service - Interactive Circuit Sandbox with Real-Time Validation
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import create_engine, Column, String, DateTime, Integer, JSON, Boolean, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import numpy as np
import uuid
import os
from datetime import datetime
import json

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/learning_companion")

# Database Setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="Lab Service - Circuit Sandbox", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Database Models =============

class LabSession(Base):
    __tablename__ = "lab_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    lab_type = Column(String(50), nullable=False)
    lab_template_id = Column(UUID(as_uuid=True))
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    final_score = Column(DECIMAL(5, 2))
    state_snapshot = Column(JSON, default={})
    mastery_achieved = Column(Boolean, default=False)

class LabAction(Base):
    __tablename__ = "lab_actions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action_type = Column(String(100), nullable=False)
    action_data = Column(JSON)
    validation_result = Column(JSON)

class LabCheckpoint(Base):
    __tablename__ = "lab_checkpoints"
    
    session_id = Column(UUID(as_uuid=True), primary_key=True)
    checkpoint_number = Column(Integer, primary_key=True)
    achieved_at = Column(DateTime, default=datetime.utcnow)
    mastery_metrics = Column(JSON)

# ============= Pydantic Models =============

class Component(BaseModel):
    id: str
    type: str  # resistor, capacitor, inductor, voltage_source, current_source, wire
    value: float = 0
    unit: str = ""
    position: Dict[str, float]  # {x, y}
    connections: List[str] = []  # List of node IDs
    orientation: int = 0  # 0, 90, 180, 270 degrees

class Node(BaseModel):
    id: str
    position: Dict[str, float]
    voltage: Optional[float] = None
    connected_components: List[str] = []

class Circuit(BaseModel):
    components: List[Component] = []
    nodes: List[Node] = []
    ground_node: Optional[str] = None

class ValidationRequest(BaseModel):
    circuit: Circuit
    validation_type: str  # kcl, kvl, power, complete

class MeasurementRequest(BaseModel):
    circuit: Circuit
    measurement_type: str  # voltage, current, power, resistance
    nodes: List[str] = []
    component_id: Optional[str] = None

class LabStartRequest(BaseModel):
    lab_type: str
    template_id: Optional[str] = None

class ComponentPlacementRequest(BaseModel):
    component: Component

# ============= Circuit Analysis Engine =============

class CircuitAnalyzer:
    """Circuit analysis and validation engine"""
    
    TOLERANCE = 1e-3  # Voltage/current tolerance for validation
    
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.node_map = {node.id: idx for idx, node in enumerate(circuit.nodes)}
        self.ground_idx = self.node_map.get(circuit.ground_node, 0)
    
    def validate_connections(self) -> Dict[str, Any]:
        """Validate that components are properly connected"""
        errors = []
        warnings = []
        
        for component in self.circuit.components:
            if component.type in ["resistor", "capacitor", "inductor"]:
                # Two-terminal components need exactly 2 connections
                if len(component.connections) != 2:
                    errors.append({
                        "component_id": component.id,
                        "error": f"{component.type} must connect to exactly 2 nodes",
                        "hint": "Check your connections"
                    })
                
                # Check if connected to same node (short circuit)
                if len(component.connections) == 2 and component.connections[0] == component.connections[1]:
                    errors.append({
                        "component_id": component.id,
                        "error": "Component creates a short circuit",
                        "hint": "Both terminals connect to the same node"
                    })
            
            elif component.type in ["voltage_source", "current_source"]:
                if len(component.connections) != 2:
                    errors.append({
                        "component_id": component.id,
                        "error": "Source must have 2 connections",
                        "hint": "Connect positive and negative terminals"
                    })
        
        # Check for ground connection
        if not self.circuit.ground_node:
            warnings.append({
                "warning": "No ground node specified",
                "hint": "Designate a reference node as ground"
            })
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def validate_kcl(self, node_id: str) -> Dict[str, Any]:
        """Validate Kirchhoff's Current Law at a node"""
        
        # Find all components connected to this node
        connected_comps = [
            comp for comp in self.circuit.components
            if node_id in comp.connections
        ]
        
        if not connected_comps:
            return {
                "valid": True,
                "node_id": node_id,
                "message": "No components connected"
            }
        
        # Calculate currents (simplified - assumes we have current values)
        # In real implementation, would solve circuit equations
        currents_in = 0
        currents_out = 0
        
        for comp in connected_comps:
            # Mock current calculation
            if comp.type == "current_source":
                if comp.connections[0] == node_id:
                    currents_in += comp.value
                else:
                    currents_out += comp.value
        
        residual = abs(currents_in - currents_out)
        
        return {
            "valid": residual < self.TOLERANCE,
            "node_id": node_id,
            "currents_in": currents_in,
            "currents_out": currents_out,
            "residual": residual,
            "message": "KCL satisfied" if residual < self.TOLERANCE else "KCL violated",
            "hint": "Sum of currents entering should equal sum leaving" if residual >= self.TOLERANCE else None
        }
    
    def validate_kvl(self, loop: List[str]) -> Dict[str, Any]:
        """Validate Kirchhoff's Voltage Law around a loop"""
        
        if len(loop) < 3:
            return {
                "valid": False,
                "message": "Loop must have at least 3 nodes"
            }
        
        # Find components in the loop
        voltage_sum = 0
        loop_components = []
        
        for i in range(len(loop)):
            node1 = loop[i]
            node2 = loop[(i + 1) % len(loop)]
            
            # Find component connecting these nodes
            for comp in self.circuit.components:
                if node1 in comp.connections and node2 in comp.connections:
                    loop_components.append(comp)
                    
                    # Add voltage drop (simplified)
                    if comp.type == "voltage_source":
                        # Check polarity
                        if comp.connections[0] == node1:
                            voltage_sum += comp.value
                        else:
                            voltage_sum -= comp.value
                    elif comp.type == "resistor":
                        # V = IR (would need current calculation)
                        pass
        
        residual = abs(voltage_sum)
        
        return {
            "valid": residual < self.TOLERANCE,
            "loop": loop,
            "components": [c.id for c in loop_components],
            "voltage_sum": voltage_sum,
            "residual": residual,
            "message": "KVL satisfied" if residual < self.TOLERANCE else "KVL violated",
            "hint": "Voltage sum around closed loop should be zero" if residual >= self.TOLERANCE else None
        }
    
    def solve_circuit(self) -> Dict[str, Any]:
        """Solve circuit using nodal analysis (simplified)"""
        
        num_nodes = len(self.circuit.nodes)
        
        if num_nodes == 0:
            return {"solved": False, "error": "No nodes in circuit"}
        
        # Build conductance matrix (simplified)
        G = np.zeros((num_nodes, num_nodes))
        I = np.zeros(num_nodes)
        
        # Process resistors
        for comp in self.circuit.components:
            if comp.type == "resistor" and len(comp.connections) == 2:
                n1 = self.node_map[comp.connections[0]]
                n2 = self.node_map[comp.connections[1]]
                
                if comp.value > 0:
                    g = 1.0 / comp.value  # Conductance
                    
                    G[n1, n1] += g
                    G[n2, n2] += g
                    G[n1, n2] -= g
                    G[n2, n1] -= g
            
            elif comp.type == "current_source" and len(comp.connections) == 2:
                n1 = self.node_map[comp.connections[0]]
                n2 = self.node_map[comp.connections[1]]
                
                I[n1] += comp.value
                I[n2] -= comp.value
        
        # Set ground node voltage to 0
        G[self.ground_idx, :] = 0
        G[self.ground_idx, self.ground_idx] = 1
        I[self.ground_idx] = 0
        
        try:
            # Solve for node voltages
            V = np.linalg.solve(G, I)
            
            # Update node voltages
            for node, voltage in zip(self.circuit.nodes, V):
                node.voltage = float(voltage)
            
            return {
                "solved": True,
                "node_voltages": {
                    node.id: float(V[idx])
                    for node, idx in zip(self.circuit.nodes, range(num_nodes))
                }
            }
        except np.linalg.LinAlgError:
            return {
                "solved": False,
                "error": "Circuit cannot be solved (singular matrix)",
                "hint": "Check for floating nodes or improper ground connection"
            }
    
    def calculate_power(self) -> Dict[str, Any]:
        """Calculate power dissipation in each component"""
        
        power_data = {}
        
        for comp in self.circuit.components:
            if comp.type == "resistor" and len(comp.connections) == 2:
                n1 = comp.connections[0]
                n2 = comp.connections[1]
                
                # Get node voltages
                v1 = next((n.voltage for n in self.circuit.nodes if n.id == n1), 0)
                v2 = next((n.voltage for n in self.circuit.nodes if n.id == n2), 0)
                
                if v1 is not None and v2 is not None:
                    voltage_drop = abs(v1 - v2)
                    if comp.value > 0:
                        power = (voltage_drop ** 2) / comp.value
                        power_data[comp.id] = {
                            "power_watts": power,
                            "voltage_drop": voltage_drop,
                            "current": voltage_drop / comp.value
                        }
        
        total_power = sum(p["power_watts"] for p in power_data.values())
        
        return {
            "components": power_data,
            "total_power_dissipation": total_power
        }

# ============= Dependencies =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= Routes =============

@app.post("/api/v1/labs/start")
async def start_lab(request: LabStartRequest, db: Session = Depends(get_db)):
    """Start a new lab session"""
    
    user_id = uuid.uuid4()  # Mock - would come from auth
    
    session = LabSession(
        user_id=user_id,
        lab_type=request.lab_type,
        lab_template_id=uuid.UUID(request.template_id) if request.template_id else None
    )
    db.add(session)
    db.commit()
    
    # Load template if provided
    initial_circuit = Circuit()
    if request.template_id:
        # In production, load from database
        pass
    
    return {
        "session_id": str(session.id),
        "lab_type": request.lab_type,
        "started_at": session.started_at.isoformat(),
        "initial_circuit": initial_circuit.dict()
    }

@app.post("/api/v1/labs/{session_id}/validate")
async def validate_circuit(
    session_id: str,
    request: ValidationRequest,
    db: Session = Depends(get_db)
):
    """Validate circuit configuration"""
    
    analyzer = CircuitAnalyzer(request.circuit)
    
    validation_result = {}
    
    if request.validation_type == "connections":
        validation_result = analyzer.validate_connections()
    
    elif request.validation_type == "kcl":
        # Validate KCL at all nodes
        kcl_results = []
        for node in request.circuit.nodes:
            result = analyzer.validate_kcl(node.id)
            kcl_results.append(result)
        
        validation_result = {
            "type": "kcl",
            "nodes": kcl_results,
            "all_valid": all(r["valid"] for r in kcl_results)
        }
    
    elif request.validation_type == "kvl":
        # For demo, validate a simple loop if ground exists
        if request.circuit.ground_node and len(request.circuit.nodes) >= 3:
            loop = [n.id for n in request.circuit.nodes[:3]]
            validation_result = analyzer.validate_kvl(loop)
        else:
            validation_result = {
                "valid": False,
                "message": "Need at least 3 nodes with ground to validate KVL"
            }
    
    elif request.validation_type == "power":
        # First solve circuit
        solve_result = analyzer.solve_circuit()
        if solve_result["solved"]:
            validation_result = analyzer.calculate_power()
            validation_result["circuit_solved"] = True
        else:
            validation_result = solve_result
    
    elif request.validation_type == "complete":
        # Run all validations
        connections = analyzer.validate_connections()
        solve_result = analyzer.solve_circuit()
        
        validation_result = {
            "connections": connections,
            "circuit_solution": solve_result,
            "all_valid": connections["valid"] and solve_result.get("solved", False)
        }
    
    # Log action
    action = LabAction(
        session_id=uuid.UUID(session_id),
        action_type="validate",
        action_data={"validation_type": request.validation_type},
        validation_result=validation_result
    )
    db.add(action)
    db.commit()
    
    return validation_result

@app.post("/api/v1/labs/{session_id}/measure")
async def measure_circuit(
    session_id: str,
    request: MeasurementRequest,
    db: Session = Depends(get_db)
):
    """Take measurements in the circuit"""
    
    analyzer = CircuitAnalyzer(request.circuit)
    
    # Solve circuit first
    solve_result = analyzer.solve_circuit()
    
    if not solve_result["solved"]:
        return {
            "error": "Cannot measure - circuit not solvable",
            "details": solve_result
        }
    
    measurements = {}
    
    if request.measurement_type == "voltage":
        # Measure voltage between nodes
        if len(request.nodes) >= 2:
            n1_voltage = solve_result["node_voltages"].get(request.nodes[0], 0)
            n2_voltage = solve_result["node_voltages"].get(request.nodes[1], 0)
            measurements = {
                "type": "voltage",
                "node1": request.nodes[0],
                "node2": request.nodes[1],
                "voltage_v": abs(n1_voltage - n2_voltage),
                "node1_to_ground": n1_voltage,
                "node2_to_ground": n2_voltage
            }
        else:
            # Measure voltage to ground
            node_voltage = solve_result["node_voltages"].get(request.nodes[0], 0)
            measurements = {
                "type": "voltage_to_ground",
                "node": request.nodes[0],
                "voltage_v": node_voltage
            }
    
    elif request.measurement_type == "current":
        # Measure current through component
        if request.component_id:
            component = next(
                (c for c in request.circuit.components if c.id == request.component_id),
                None
            )
            if component and len(component.connections) == 2:
                n1 = component.connections[0]
                n2 = component.connections[1]
                v1 = solve_result["node_voltages"].get(n1, 0)
                v2 = solve_result["node_voltages"].get(n2, 0)
                
                if component.type == "resistor" and component.value > 0:
                    current = abs(v1 - v2) / component.value
                    measurements = {
                        "type": "current",
                        "component_id": request.component_id,
                        "current_a": current,
                        "voltage_drop": abs(v1 - v2)
                    }
    
    elif request.measurement_type == "power":
        power_data = analyzer.calculate_power()
        measurements = power_data
    
    # Log measurement
    action = LabAction(
        session_id=uuid.UUID(session_id),
        action_type="measure",
        action_data={"measurement_type": request.measurement_type},
        validation_result=measurements
    )
    db.add(action)
    db.commit()
    
    return measurements

@app.post("/api/v1/labs/{session_id}/save-state")
async def save_circuit_state(
    session_id: str,
    circuit: Circuit,
    db: Session = Depends(get_db)
):
    """Save current circuit state"""
    
    session = db.query(LabSession).filter(LabSession.id == uuid.UUID(session_id)).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.state_snapshot = circuit.dict()
    db.commit()
    
    return {"success": True, "session_id": session_id}

@app.post("/api/v1/labs/{session_id}/checkpoint")
async def achieve_checkpoint(
    session_id: str,
    checkpoint_number: int,
    mastery_metrics: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Record achievement of a checkpoint"""
    
    checkpoint = LabCheckpoint(
        session_id=uuid.UUID(session_id),
        checkpoint_number=checkpoint_number,
        mastery_metrics=mastery_metrics
    )
    db.add(checkpoint)
    db.commit()
    
    return {
        "success": True,
        "checkpoint": checkpoint_number,
        "achieved_at": checkpoint.achieved_at.isoformat()
    }

@app.post("/api/v1/labs/{session_id}/complete")
async def complete_lab(
    session_id: str,
    final_score: float,
    mastery_achieved: bool,
    db: Session = Depends(get_db)
):
    """Complete a lab session"""
    
    session = db.query(LabSession).filter(LabSession.id == uuid.UUID(session_id)).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.completed_at = datetime.utcnow()
    session.final_score = final_score
    session.mastery_achieved = mastery_achieved
    db.commit()
    
    return {
        "session_id": session_id,
        "final_score": final_score,
        "mastery_achieved": mastery_achieved,
        "duration_seconds": int((session.completed_at - session.started_at).total_seconds())
    }

@app.get("/api/v1/labs/{session_id}/actions")
async def get_lab_actions(session_id: str, db: Session = Depends(get_db)):
    """Get all actions in a lab session"""
    
    actions = db.query(LabAction)\
        .filter(LabAction.session_id == uuid.UUID(session_id))\
        .order_by(LabAction.timestamp)\
        .all()
    
    return {
        "session_id": session_id,
        "actions": [
            {
                "id": str(a.id),
                "type": a.action_type,
                "timestamp": a.timestamp.isoformat(),
                "data": a.action_data,
                "result": a.validation_result
            }
            for a in actions
        ]
    }

# ============= WebSocket for Real-Time Lab Interaction =============

class LabConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

lab_manager = LabConnectionManager()

@app.websocket("/ws/labs/{session_id}")
async def lab_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time lab interaction"""
    
    await lab_manager.connect(session_id, websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            action_type = data.get("type")
            
            if action_type == "place_component":
                component_data = data.get("component")
                component = Component(**component_data)
                
                # Validate placement
                response = {
                    "type": "component_placed",
                    "component_id": component.id,
                    "valid": True,
                    "message": f"{component.type} placed successfully"
                }
                
                await lab_manager.send(session_id, response)
            
            elif action_type == "validate_live":
                circuit_data = data.get("circuit")
                circuit = Circuit(**circuit_data)
                analyzer = CircuitAnalyzer(circuit)
                
                # Run quick validation
                connections = analyzer.validate_connections()
                
                response = {
                    "type": "validation_result",
                    "result": connections,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await lab_manager.send(session_id, response)
            
            elif action_type == "measure_live":
                circuit_data = data.get("circuit")
                circuit = Circuit(**circuit_data)
                measurement_type = data.get("measurement_type")
                nodes = data.get("nodes", [])
                
                analyzer = CircuitAnalyzer(circuit)
                solve_result = analyzer.solve_circuit()
                
                if solve_result["solved"]:
                    measurements = {
                        "type": "measurement_result",
                        "measurement_type": measurement_type,
                        "values": solve_result["node_voltages"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    measurements = {
                        "type": "measurement_error",
                        "error": solve_result.get("error", "Cannot measure"),
                        "hint": solve_result.get("hint")
                    }
                
                await lab_manager.send(session_id, measurements)
    
    except WebSocketDisconnect:
        lab_manager.disconnect(session_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "lab-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)