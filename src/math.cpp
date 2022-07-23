#include <stdexcept>
#include <cmath>

#include "math.h"

Matr matMult(const Matr& a, const Matr& b)
{
    if (a.empty() || b.empty())
        throw std::runtime_error("matMult: Empty vector");

    const int aRows = a.size();
    const int aCols = a[0].size();
    const int bRows = b.size();
    const int bCols = b[0].size();
    if (aCols != bRows)
        throw std::runtime_error("matMult: vector 1 column must be equal to vector 2 row");

    Matr result(aRows, std::vector<float>(bCols));
    for(int i = 0; i < aRows; ++i)
        for(int j = 0; j < bCols; ++j)
            for(int k = 0; k < aCols; ++k)
                result[i][j] += a[i][k] * b[k][j];
    return result;
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("cosineSimilarity: vector dimentions must be equal");

    double dot = 0.0f;
    double denomA = 0.0f;
    double denomB = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
        denomA += a[i] * a[i];
        denomB += b[i] * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(denomA) * std::sqrt(denomB) + 1e-6));
}

std::pair<int, float> searchMostSimilarEmbedding(
    const Matr& embeddings, const std::vector<float>& newComerEmbedding)
{
    if (embeddings.empty() || newComerEmbedding.empty())
        throw std::runtime_error("searchMostSimilarEmbedding: Empty vector");

    int bestId = 0;
    float bestSim = -1.0f;
    for (std::size_t i = 0; i < embeddings.size(); ++i)
    {
        const auto& embedding = embeddings.at(i);
        const auto cosim = cosineSimilarity(embedding, newComerEmbedding);
        if (cosim > bestSim)
        {
            bestSim = cosim;
            bestId = i;
        }
    }
    return {bestId, bestSim};
}

std::vector<float> avgEmbedding(const Matr& embeddings)
{
    const int nEmbeddings = embeddings.size();
    if (0 == nEmbeddings)
        throw std::runtime_error("avgEmbedding: Empty vector");

    if (1 == nEmbeddings)
        return embeddings[0];

    const int embeddingDim = embeddings[0].size();

    std::vector<float> result(embeddingDim, 0.0f);
    for (const auto& embedding : embeddings)
        for (std::size_t i = 0; i < embeddingDim; ++i)
            result[i] += embedding[i];
    for (std::size_t i = 0; i < embeddingDim; ++i)
        result[i] /= nEmbeddings;
    return result;
}