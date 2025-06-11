from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

circuit = QuantumCircuit(2)
circuit.measure_all()

service = QiskitRuntimeService(
    token='', #your_api_token_key
    channel='ibm_quantum' #your_channel_name
)
backend = service.least_busy(operational=True, simulator=False)

sampler = Sampler(backend)
job = sampler.run([circuit])
result = job.result()

print(result)