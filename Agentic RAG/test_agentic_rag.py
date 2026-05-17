
import pytest
import requests
import json
import time
from vector_search import VectorDBSearchTool
from local_llm import LocalOllamaLLM

# ============================================
# UNIT TESTS - Test individual components
# ============================================

class TestVectorDBSearchTool:
    """Test the vector database search functionality"""
    
    def test_initialization(self):
        """Test if VectorDBSearchTool initializes correctly"""
        tool = VectorDBSearchTool()
        assert tool.dimension == 768
        assert tool.index is not None
        assert len(tool.documents) == 0
    
    def test_add_documents(self):
        """Test adding documents to the vector store"""
        tool = VectorDBSearchTool()
        test_docs = [
            "Artificial intelligence is transforming the world.",
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks."
        ]
        tool.add_docs(test_docs)
        assert len(tool.documents) == 3
    
    def test_search_returns_results(self):
        """Test searching returns relevant results"""
        tool = VectorDBSearchTool()
        test_docs = [
            "Python is a programming language.",
            "Java is also a programming language.",
            "Apples are healthy fruits."
        ]
        tool.add_docs(test_docs)
        results = tool("What programming languages exist?", k=2)
        assert len(results) == 2
        assert "document" in results[0]
        assert "score" in results[0]
    
    def test_search_with_empty_index(self):
        """Test searching an empty index"""
        tool = VectorDBSearchTool()
        results = tool("test query")
        print(results)
        assert len(results) == 0 or all(r["document"] == "" for r in results)


class TestLocalOllamaLLM:
    """Test the local LLM integration"""
    
    def test_initialization(self):
        """Test if LocalOllamaLLM initializes"""
        llm = LocalOllamaLLM()
        assert llm.client is not None
        assert llm.model is not None
    
    def test_call_method_exists(self):
        """Test if call method exists and is callable"""
        llm = LocalOllamaLLM()
        assert callable(getattr(llm, 'call', None))
    
    # Note: Actual LLM calls would require Ollama to be running
    # These tests would be integration tests


class TestFireCrawlIntegration:
    """Test Firecrawl web search integration"""
    
    def test_firecrawl_import(self):
        """Test if Firecrawl is properly imported"""
        from fire_crawl import fc
        assert fc is not None
    
    def test_firecrawl_search(self):
        """Test Firecrawl search functionality (requires API key)"""
        from fire_crawl import FireCrawl
        fc = FireCrawl()
        try:
            result = fc("test query")
            assert result is not None
        except Exception as e:
            pytest.skip(f"Firecrawl API not available: {e}")


# ============================================
# INTEGRATION TESTS - Test the full API
# ============================================

class TestAgenticRAGAPIIntegration:
    """Integration tests for the complete AgenticRAG API"""
    
    @pytest.fixture
    def api_server(self):
        """Start the API server for testing"""
        import subprocess
        import sys
        
        # Start server in background
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        yield process
        
        # Cleanup
        process.terminate()
        process.wait()
    
    def test_server_health_check(self, api_server):
        """Test if the server is running"""
        try:
            response = requests.get("http://localhost:8000/health")
            assert response.status_code == 200
        except requests.ConnectionError:
            pytest.fail("Server is not running")
    
    def test_api_prediction_endpoint(self, api_server):
        """Test the main prediction endpoint"""
        test_payload = {
            "query": "What is artificial intelligence?"
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_payload
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data or "response" in data


# ============================================
# FUNCTIONAL TESTS - Test real-world scenarios
# ============================================

class TestFunctionalScenarios:
    """Test real-world usage scenarios"""
    
    def test_research_query(self):
        """Test a research-style query"""
        # This would test the full agent workflow
        pass
    
    def test_vector_search_accuracy(self):
        """Test that vector search returns semantically similar documents"""
        tool = VectorDBSearchTool()
        tool.add_docs([
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning algorithms learn from data.",
            "Python is great for data science."
        ])
        
        results = tool("animals and pets", k=1)
        # Should return the fox/dog sentence as most relevant
        assert any("fox" in r["document"].lower() or "dog" in r["document"].lower() 
                  for r in results)


# ============================================
# ERROR HANDLING TESTS
# ============================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_query(self):
        """Test handling of empty query"""
        tool = VectorDBSearchTool()
        results = tool("")
        # Should handle gracefully
    
    def test_invalid_k_parameter(self):
        """Test invalid k parameter in search"""
        tool = VectorDBSearchTool()
        tool.add_docs(["test document"])
        results = tool("test", k=-1)
        # Should handle or raise appropriate error
    
    def test_large_k_parameter(self):
        """Test k larger than available documents"""
        tool = VectorDBSearchTool()
        tool.add_docs(["doc1", "doc2"])
        results = tool("test", k=100)
        # Should return only available documents


# ============================================
# PERFORMANCE TESTS
# ============================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_search_speed(self):
        """Test that search completes in reasonable time"""
        tool = VectorDBSearchTool()
        # Add many documents
        docs = [f"Document number {i} with some content." for i in range(1000)]
        tool.add_docs(docs)
        
        start_time = time.time()
        results = tool("test query")
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0  # Should complete within 5 seconds


if __name__ == "__main__":
    pass
    # pytest.main([__file__, "-v"])


'''
#!/bin/bash

echo "=== Agentic RAG Test Suite ==="

# Run unit tests
echo "Running unit tests..."
python -m pytest test_agentic_rag.py::TestVectorDBSearchTool -v

echo "Running LLM tests..."
python -m pytest test_agentic_rag.py::TestLocalOllamaLLM -v

echo "Running integration tests..."
python -m pytest test_agentic_rag.py::TestAgenticRAGAPIIntegration -v

echo "=== Tests Complete ==="

'''