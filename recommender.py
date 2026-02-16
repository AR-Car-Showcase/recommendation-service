import json
import fcntl
import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class Recommender:
    def __init__(self, data_path="dataset/cars_data_final.json", bandit_path="bandit_state.json",
                 cache_dir="cache"):
        self.data_path = data_path
        self.bandit_path = bandit_path
        self.cache_dir = cache_dir
        self.df = None
        self.similarity_matrix = None
        self.car_indices = None
        self.successes = defaultdict(int)
        self.failures = defaultdict(int)
        self.pending_updates = 0
        self.update_threshold = 10

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.load_data()
        self._load_bandit_state()

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        with open(self.data_path, 'r') as f:
            data = json.load(f)

        cars = []
        for car in data:
            brand = car.get("brand", "").lower()
            model = car.get("model", "").lower()
            base_info = {
                "car_id": f"{brand}_{model}",
                "brand": brand,
                "model": model,
                "body_type": car.get("body_type", "").lower(),
                "fuel_type": car.get("fuel_type", "").lower(),
                "transmission_type": car.get("transmission_type", "").lower(),
                "price": car.get("min_price_lakhs", 0),
                "rating": car.get("rating", 0),
            }
            cars.append(base_info)

        self.df = pd.DataFrame(cars)
        self._compute_similarity()

    def _compute_similarity(self):
        cache_matrix_path = os.path.join(self.cache_dir, "similarity_matrix.npy")
        cache_indices_path = os.path.join(self.cache_dir, "car_indices.json")

        # Check if cache is valid (exists and newer than data file)
        if os.path.exists(cache_matrix_path) and os.path.exists(cache_indices_path):
            data_mtime = os.path.getmtime(self.data_path)
            cache_mtime = os.path.getmtime(cache_matrix_path)

            if cache_mtime > data_mtime:
                try:
                    self.similarity_matrix = np.load(cache_matrix_path)
                    with open(cache_indices_path, 'r') as f:
                        self.car_indices = json.load(f)
                    return
                except Exception as e:
                    print(f"Error loading cache: {e}. Recomputing...")

        # Recompute if cache invalid or missing
        scaler = MinMaxScaler()
        numerical_features = scaler.fit_transform(self.df[['price', 'rating']])

        def create_soup(x):
            return f"{x['brand']} {x['body_type']} {x['fuel_type']} {x['transmission_type']}"

        self.df['soup'] = self.df.apply(create_soup, axis=1)

        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df['soup'])

        cosine_sim_categorical = cosine_similarity(count_matrix, count_matrix)
        cosine_sim_numerical = cosine_similarity(numerical_features, numerical_features)

        self.similarity_matrix = (0.7 * cosine_sim_categorical) + (0.3 * cosine_sim_numerical)
        self.car_indices = pd.Series(self.df.index, index=self.df['model']).to_dict()

        # Save to cache
        try:
            np.save(cache_matrix_path, self.similarity_matrix)
            with open(cache_indices_path, 'w') as f:
                json.dump(self.car_indices, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _load_bandit_state(self):
        if not os.path.exists(self.bandit_path):
            self.successes = defaultdict(int)
            self.failures = defaultdict(int)
            return
        try:
            with open(self.bandit_path, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                    self.successes = defaultdict(int, {k: int(v) for k, v in data.get('successes', {}).items()})
                    self.failures = defaultdict(int, {k: int(v) for k, v in data.get('failures', {}).items()})
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (json.JSONDecodeError, IOError):
            self.successes = defaultdict(int)
            self.failures = defaultdict(int)

    def _save_bandit_state(self):
        try:
            mode = 'r+' if os.path.exists(self.bandit_path) else 'w'
            with open(self.bandit_path, mode) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    f.truncate()
                    json.dump({
                        'successes': dict(self.successes),
                        'failures': dict(self.failures)
                    }, f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def _softmax(self, scores, temperature=0.7):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted / max(temperature, 0.01))
        return exp_scores / exp_scores.sum()

    def update_feedback(self, car_id: str, positive: bool):
        car_id = str(car_id).lower()
        if positive:
            self.successes[car_id] += 1
        else:
            self.failures[car_id] += 1
        
        self.pending_updates += 1
        if self.pending_updates >= self.update_threshold:
            self._save_bandit_state()
            self.pending_updates = 0

    def get_similar_cars(self, car_model_name: str, top_n: int = 5):
        car_model_name = car_model_name.lower()
        if car_model_name not in self.car_indices:
            matches = [m for m in self.car_indices.keys() if car_model_name in m]
            if not matches:
                return []
            car_model_name = matches[0]

        idx = self.car_indices[car_model_name]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        car_indices = [i[0] for i in sim_scores]

        return self.df.iloc[car_indices][['brand', 'model']].to_dict('records')

    def get_personalized(self, seed_car_models: list, top_n: int = 5):
        valid_indices = []
        for model in seed_car_models:
            model_lower = model.lower()
            if model_lower in self.car_indices:
                valid_indices.append(self.car_indices[model_lower])
            else:
                matches = [m for m in self.car_indices.keys() if model_lower in m]
                if matches:
                    valid_indices.append(self.car_indices[matches[0]])

        if not valid_indices:
            return []

        seed_vectors = self.similarity_matrix[valid_indices]
        mean_vector = np.mean(seed_vectors, axis=0)
        sorted_indices = np.argsort(mean_vector)[::-1]

        recommendations = []
        for idx in sorted_indices:
            if idx not in valid_indices:
                recommendations.append(idx)
                if len(recommendations) >= top_n:
                    break

        return self.df.iloc[recommendations][['brand', 'model']].to_dict('records')

    def get_personalized_ranking(self, preferences: dict, top_n: int = 5,
                                 exploration_rate: float = 0.3):
        user_id = preferences.get('user_id', 'default')
        # Use a consistent seed for the user/session to avoid jumpy results
        seed_components = [user_id, str(datetime.now().hour), str(datetime.now().day)]
        session_id = preferences.get('session_id')
        if session_id:
            seed_components.append(session_id)
        
        user_seed = hash("_".join(seed_components)) % (2 ** 32)
        np.random.seed(user_seed)

        exclude_ids = set(str(eid).lower() for eid in preferences.get('exclude_car_ids', []))

        # Filter once, then work on the slice
        if exclude_ids:
            scored_df = self.df[~self.df['car_id'].isin(exclude_ids)].copy()
        else:
            scored_df = self.df.copy()

        if len(scored_df) < top_n:
            # If we filtered out too many, bring some back to fulfill top_n
            repeats_needed = top_n - len(scored_df)
            repeat_cars = self.df[self.df['car_id'].isin(exclude_ids)].head(repeats_needed)
            scored_df = pd.concat([scored_df, repeat_cars])

        if scored_df.empty:
            return []

        # Vectorized scoring
        scores = np.zeros(len(scored_df))
        
        pref_weights = {
            'brands': 3.0,
            'body_types': 2.0,
            'fuel_types': 1.5,
            'transmissions': 1.5
        }
        
        attr_map = {
            'brands': 'brand',
            'body_types': 'body_type',
            'fuel_types': 'fuel_type',
            'transmissions': 'transmission_type'
        }

        for pref_key, weight in pref_weights.items():
            vals = preferences.get(pref_key)
            if vals:
                vals_lower = [v.lower() for v in vals]
                scores += scored_df[attr_map[pref_key]].isin(vals_lower).astype(float) * weight

        budget = preferences.get('budget')
        if budget:
            scores += (scored_df['price'] <= budget).astype(float) * 1.5

        scored_df['score'] = scores
        scored_df = self._apply_thompson_sampling(scored_df)

        # Sort by final score
        scored_df = scored_df.sort_values(by=['final_score', 'rating'], ascending=[False, False])

        if np.random.random() < exploration_rate:
            return self._explore_recommendations(scored_df, top_n)
        else:
            return self._exploit_recommendations(scored_df, top_n)

    def _apply_thompson_sampling(self, scored_df):
        thompson_multipliers = []
        for _, car in scored_df.iterrows():
            car_id = str(car['car_id'])
            alpha = self.successes[car_id] + 1
            beta_val = self.failures[car_id] + 1
            total_samples = (alpha - 1) + (beta_val - 1)
            exploration_bonus = 1.0 / (1 + total_samples)
            sampled_reward = np.random.beta(alpha, beta_val) + exploration_bonus * 0.2
            sampled_reward = min(sampled_reward, 1.0)
            multiplier = 1.0 + (sampled_reward * 0.3)
            thompson_multipliers.append(multiplier)

        scored_df = scored_df.copy()
        scored_df['thompson_multiplier'] = thompson_multipliers
        scored_df['final_score'] = scored_df['score'] * scored_df['thompson_multiplier']
        return scored_df

    def _exploit_recommendations(self, scored_df, top_n):
        top_cars = scored_df.head(top_n)
        return top_cars[['brand', 'model']].to_dict('records')

    def _explore_recommendations(self, scored_df, top_n):
        candidate_pool = min(top_n * 4, len(scored_df))
        candidates = scored_df.head(candidate_pool)

        probabilities = self._softmax(candidates['final_score'].values, temperature=0.7)

        sample_size = min(top_n, len(candidates))
        selected_indices = np.random.choice(
            candidates.index,
            size=sample_size,
            replace=False,
            p=probabilities
        )

        result = scored_df.loc[selected_indices]
        result = result.sort_values(by='final_score', ascending=False)
        return result[['brand', 'model']].to_dict('records')


if __name__ == "__main__":
    rec = Recommender()
    print(rec.get_similar_cars("verna", 5))
    print(rec.get_personalized_ranking({
        "brands": ["bmw", "audi"],
        "body_types": ["suv"],
        "fuel_types": ["petrol"],
        "transmissions": ["automatic"],
        "budget": 80.0,
        "user_id": "test_user_1"
    }))
