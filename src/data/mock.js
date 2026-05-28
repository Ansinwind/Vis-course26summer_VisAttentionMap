// 3 models × 10 samples mock data

function randWeights(rows, cols) {
  return Array.from({ length: rows }, () => {
    const row = Array.from({ length: cols }, () => Math.random())
    const sum = row.reduce((a, b) => a + b, 0)
    return row.map(v => v / sum)
  })
}

const MODELS = ['CLIP', 'BLIP2', 'LLaVA']
const SAMPLES = Array.from({ length: 10 }, (_, i) => `s${String(i + 1).padStart(3, '0')}`)
const QUESTIONS = [
  'What color is the dog?',
  'How many people are in the image?',
  'What is the man holding?',
  'Is it daytime or nighttime?',
  'What sport is being played?',
  'What is on the table?',
  'Where is the cat sitting?',
  'What color is the car?',
  'How many birds are there?',
  'What is the woman wearing?',
]

export const matrixData = MODELS.flatMap(model =>
  SAMPLES.map((sample_id, i) => ({
    model,
    sample_id,
    correct: Math.random() > 0.4,
    confidence: parseFloat((0.3 + Math.random() * 0.7).toFixed(2)),
  }))
)

export const attentionData = Object.fromEntries(
  MODELS.flatMap(model =>
    SAMPLES.map((sample_id, i) => [
      `${model}__${sample_id}`,
      {
        image_url: `https://picsum.photos/seed/${sample_id}/224/224`,
        question: QUESTIONS[i],
        // 8 image patch tokens × 6 text tokens
        weights: randWeights(8, 6),
      },
    ])
  )
)
