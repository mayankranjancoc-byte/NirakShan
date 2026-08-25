import unittest
from modules.audit_logger import log_screening_result, get_recent_screenings, store_embedding, find_similar_identity

class TestAuditAndIdentity(unittest.TestCase):
    def test_case_1_audit_log_write(self):
        mock_result = {
            "risk": {
                "risk_score": 45.0,
                "verdict": "MEDIUM",
                "flags": ["TEST_FLAG"]
            },
            "ocr": {
                "document_type": "PASSPORT",
                "mrz_status": "VALID"
            },
            "face": {
                "verified": True
            },
            "tampering": {
                "tamper_score": 12.0,
                "breakdown": {
                    "exif_analysis": {"score": 5.0}
                }
            }
        }
        
        session_id = log_screening_result(mock_result)
        recent = get_recent_screenings(1)
        self.assertTrue(len(recent) >= 1)
        record = recent[0]
        
        self.assertEqual(record["session_id"], session_id)
        self.assertEqual(record["document_type"], "PASSPORT")
        self.assertEqual(record["risk_score"], 45.0)
        self.assertEqual(record["verdict"], "MEDIUM")
        self.assertEqual(record["face_verified"], "MATCH")
        self.assertEqual(record["exif_anomaly_score"], 5.0)
        self.assertEqual(record["tamper_score"], 12.0)
        self.assertIn("TEST_FLAG", record["flags"])
        
    def test_case_2_duplicate_identity_flag(self):
        # Store dummy embedding
        dummy_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        store_embedding("session_123", "X123456", dummy_embedding)
        
        # Search with identical embedding but different passport
        match = find_similar_identity(dummy_embedding, "Y789012")
        
        self.assertIsNotNone(match)
        self.assertEqual(match["matched_session"], "session_123")
        self.assertEqual(match["matched_passport"], "X123456")
        self.assertGreater(match["similarity"], 0.99)

if __name__ == '__main__':
    unittest.main()
