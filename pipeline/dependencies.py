from pipeline.extractor import EntityExtractor
from pipeline.risk_engine import RiskEngine
from pipeline.llm_router import LLMRouter
from pipeline.recommender import RecommendationBuilder
from pipeline.safety_validator import SafetyValidator
from rag.retriever import DiseaseRetriever

extractor_inst = EntityExtractor()
risk_engine_inst = RiskEngine()
retriever_inst = DiseaseRetriever()
llm_router_inst = LLMRouter()
recommender_inst = RecommendationBuilder()
validator_inst = SafetyValidator()
