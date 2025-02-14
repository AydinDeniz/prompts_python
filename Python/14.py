
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, average_precision_score

class RecommendationSystem:
    def __init__(self, data, user_column, item_column, rating_column):
        self.data = data
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.user_item_matrix = None
        self.similarity_matrix = None

    def preprocess_data(self):
        self.user_item_matrix = self.data.pivot_table(index=self.user_column, columns=self.item_column, values=self.rating_column).fillna(0)

    def compute_similarity(self):
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        np.fill_diagonal(self.similarity_matrix, 0)

    def recommend_items(self, user_id, top_k=5):
        user_index = self.user_item_matrix.index.get_loc(user_id)
        similarity_scores = self.similarity_matrix[user_index]
        similar_users = np.argsort(similarity_scores)[::-1]
        recommendations = {}
        for similar_user in similar_users:
            similar_user_id = self.user_item_matrix.index[similar_user]
            for item, rating in self.user_item_matrix.loc[similar_user_id].items():
                if self.user_item_matrix.loc[user_id, item] == 0 and rating > 0:
                    recommendations[item] = recommendations.get(item, 0) + similarity_scores[similar_user]
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_recommendations[:top_k]]

    def evaluate(self, test_data, top_k=5):
        y_true = []
        y_pred = []
        for user_id in test_data[self.user_column].unique():
            if user_id not in self.user_item_matrix.index:
                continue
            user_actual = set(test_data[test_data[self.user_column] == user_id][self.item_column])
            user_recommendations = set(self.recommend_items(user_id, top_k=top_k))
            y_true.append([1 if item in user_actual else 0 for item in user_recommendations])
            y_pred.append([1] * len(user_recommendations))
        precision = precision_score(np.hstack(y_true), np.hstack(y_pred), average='binary', zero_division=1)
        return precision

if __name__ == "__main__":
    data = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3],
        "item_id": ["A", "B", "C", "A", "C", "B", "D"],
        "rating": [5, 4, 3, 5, 2, 4, 5]
    })

    recommender = RecommendationSystem(data, "user_id", "item_id", "rating")
    recommender.preprocess_data()
    recommender.compute_similarity()
    print("Recommendations for user 1:", recommender.recommend_items(1, top_k=3))

    test_data = pd.DataFrame({
        "user_id": [1, 2, 3],
        "item_id": ["D", "B", "A"]
    })
    precision = recommender.evaluate(test_data, top_k=3)
    print(f"Precision@K: {precision}")
