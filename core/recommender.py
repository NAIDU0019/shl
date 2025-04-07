import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from typing import List, Optional
import logging

class ProductRecommender:
    def __init__(self, product_data_path: str = "data/shl_products_clean.csv"):
        self.logger = logging.getLogger(__name__)
        self.df = self._load_and_validate_data(product_data_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self._prepare_embeddings()
    
    def _load_and_validate_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                Path(path),
                quotechar='"',
                escapechar='\\',
                on_bad_lines='warn'
            )
            
            required_columns = {
                "Product", "Description", "Job Levels",
                "Assessment Length (minutes)", "Category", "Keywords"
            }
            
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")
            
            df = df.dropna(subset=["Product", "Description"])
            df["Keywords"] = df["Keywords"].fillna(df["Category"].str.lower())
            
            return df
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def _prepare_embeddings(self):
        self.df["embedding_text"] = (
            self.df["Product"] + " " +
            self.df["Description"] + " " +
            self.df["Category"] + " " +
            self.df["Keywords"]
        )
        
        text_chunks = [self.df["embedding_text"].tolist()[i:i+32] 
                       for i in range(0, len(self.df), 32)]
        
        embeddings = []
        for chunk in text_chunks:
            embeddings.append(self.model.encode(
                chunk,
                convert_to_tensor=True,
                show_progress_bar=False
            ))
        
        self.embeddings = torch.cat(embeddings)

    def recommend(
        self,
        input_text: str,
        experience_level: Optional[str] = None,
        preferred_categories: Optional[List[str]] = None,
        top_k: int = 5,
        min_score: float = 0.15
    ) -> pd.DataFrame:
        try:
            input_embed = self.model.encode(input_text, convert_to_tensor=True)
            with torch.no_grad():
                scores = util.cos_sim(input_embed, self.embeddings)[0].cpu().numpy()
            
            results = self.df.copy()
            results["score"] = scores
            
            if experience_level and experience_level.lower() != "any":
                results = results[
                    results["Job Levels"].str.contains(
                        experience_level,
                        case=False,
                        na=False
                    )
                ]
            
            if preferred_categories:
                results = results[results["Category"].isin(preferred_categories)]
            
            filtered = results[results["score"] >= min_score]
            
            if len(filtered) >= top_k:
                return filtered.sort_values("score", ascending=False).head(top_k)
            
            if len(filtered) < top_k and min_score > 0.1:
                return self.recommend(
                    input_text,
                    experience_level,
                    preferred_categories,
                    top_k,
                    min_score=0.1
                )
            
            keywords = '|'.join(input_text.lower().split()[:3])
            keyword_matches = self.df[
                self.df["Keywords"].str.contains(keywords, case=False, na=False)
            ]
            if len(keyword_matches) > 0:
                keyword_matches = keyword_matches.copy()
                keyword_matches["score"] = 0.4
                return keyword_matches.sort_values("score", ascending=False).head(top_k)
            
            return results.sort_values("score", ascending=False).head(top_k)
        
        except Exception as e:
            self.logger.error(f"Recommendation failed: {str(e)}")
            return pd.DataFrame(columns=self.df.columns.tolist() + ["score"])
