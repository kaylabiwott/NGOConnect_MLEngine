package com.ngoconnect.app

import android.content.Context
import com.google.gson.Gson
import kotlin.math.exp
import kotlin.math.sqrt

// The trained model, exported from the notebook as numbers (assets/ml_model.json)
data class ModelData(
    val classes: List<String>,
    val vocabulary: List<String>,
    val idf: List<Double>,
    val availability: List<String>,
    val locations: List<String>,
    val coef: List<List<Double>>,
    val intercept: List<Double>
)

data class Recommendation(
    val cause: String,
    val confidence: Double,      // percentage, e.g. 89.7
    val reasons: List<String>    // why this cause was suggested
)

sealed class RecommendResult {
    data class Success(
        val recommendations: List<Recommendation>,
        val ignoredWords: List<String>,
        val lowConfidence: Boolean
    ) : RecommendResult()

    data class Error(val message: String) : RecommendResult()
}

// Same steps as the notebook: check input -> TF-IDF + one-hot -> Logistic Regression -> explain
class MlEngine(private val model: ModelData) {

    private val vocabIndex: Map<String, Int> =
        model.vocabulary.withIndex().associate { it.value to it.index }
    private val tokenRegex = Regex("\\b\\w\\w+\\b")

    val availabilityOptions: List<String> get() = model.availability
    val locationOptions: List<String> get() = model.locations

    fun recommend(
        skills: String?,
        availability: String?,
        location: String?,
        topN: Int = 3
    ): RecommendResult {
        // 1. validate the input
        if (skills.isNullOrBlank()) {
            return RecommendResult.Error("Please enter at least one skill, for example: nursing, first aid.")
        }
        val avail = model.availability.firstOrNull { it.equals(availability?.trim(), ignoreCase = true) }
            ?: return RecommendResult.Error(
                "Availability must be one of: " + model.availability.joinToString(", ") + "."
            )
        val loc = model.locations.firstOrNull { it.equals(location?.trim(), ignoreCase = true) }
            ?: return RecommendResult.Error(
                "Location must be one of: " + model.locations.joinToString(", ") + "."
            )

        val words = tokenRegex.findAll(skills.lowercase().trim()).map { it.value }.toList()
        val known = words.filter { it in vocabIndex }
        val unknown = words.filter { it !in vocabIndex }
        if (known.isEmpty()) {
            return RecommendResult.Error(
                "None of those skills were recognised. Try skills such as teaching, nursing, tree planting or cooking."
            )
        }

        // 2. turn the input into numbers: [skill words (TF-IDF) | availability | location]
        val v = model.vocabulary.size
        val a = model.availability.size
        val x = DoubleArray(v + a + model.locations.size)
        for (w in known) x[vocabIndex.getValue(w)] += 1.0
        var norm = 0.0
        for (j in 0 until v) {
            x[j] *= model.idf[j]
            norm += x[j] * x[j]
        }
        norm = sqrt(norm)
        for (j in 0 until v) x[j] /= norm
        x[v + model.availability.indexOf(avail)] = 1.0
        x[v + a + model.locations.indexOf(loc)] = 1.0

        // 3. score each cause (Logistic Regression), then turn scores into percentages
        val scores = DoubleArray(model.classes.size) { c ->
            var s = model.intercept[c]
            for (j in x.indices) s += model.coef[c][j] * x[j]
            s
        }
        val maxScore = scores.max()
        val exps = scores.map { exp(it - maxScore) }
        val total = exps.sum()
        val probs = exps.map { it / total }

        // 4. explain: weight x value for each input; the biggest positive ones are the reasons
        val results = probs.indices.sortedByDescending { probs[it] }.take(topN).map { c ->
            val confidence = Math.round(probs[c] * 1000.0) / 10.0
            val reasons = if (confidence >= 10.0) {
                x.indices
                    .map { j -> j to model.coef[c][j] * x[j] }
                    .filter { it.second > 0 }
                    .sortedByDescending { it.second }
                    .take(3)
                    .map { readable(it.first, v, a) }
            } else emptyList()
            Recommendation(model.classes[c], confidence, reasons)
        }

        return RecommendResult.Success(
            recommendations = results,
            ignoredWords = unknown,
            lowConfidence = results.first().confidence < 40.0
        )
    }

    private fun readable(j: Int, v: Int, a: Int): String = when {
        j < v -> "skill: " + model.vocabulary[j]
        j < v + a -> "availability: " + model.availability[j - v]
        else -> "location: " + model.locations[j - v - a]
    }

    companion object {
        fun load(context: Context): MlEngine {
            val json = context.assets.open("ml_model.json").bufferedReader().use { it.readText() }
            return MlEngine(Gson().fromJson(json, ModelData::class.java))
        }
    }
}
