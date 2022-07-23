#pragma once

#include <vector>
#include <utility>

using Matr = std::vector<std::vector<float>>;

Matr matMult(const Matr& a, const Matr& b);

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

std::pair<int, float> mostSimilar(
    const Matr& embeddings, const std::vector<float>& newComerEmbedding);

std::vector<float> avgEmbedding(const Matr& embeddings);