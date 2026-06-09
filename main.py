from fastapi import FastAPI
# 1) on cree leapplication. title/description alimentent la doc auto (Swagger)
app = FastAPI(title="DataProphet API", description="API de prediction de l'annee de plantation d'un arbre", version="1.0.0",)

# 2) une route GET sur l'URL /health
# le decorateur @app.get("/health") dit a FastAPI:
# "quand quels'un fait  GET /health, execute la fonction en dessous"

@app.get("/health")
def health_check():
	# ce dictionaire Python sera automatiqument converti en JSON par fastAPI
	return {"status": "OK", "message": "le service DataProphet est actif"}

