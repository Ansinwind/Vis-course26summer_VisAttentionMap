<template>
  <div style="padding:16px">
    <div v-if="!sampleId" style="color:#999;padding:40px;text-align:center">
      点击左侧矩阵中的格子查看注意力对比
    </div>
    <template v-else>
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
        <img :src="info.image_url" style="width:112px;height:112px;object-fit:cover;border-radius:4px" />
        <div>
          <div style="font-size:13px;color:#666">问题</div>
          <div style="font-weight:bold">{{ info.question }}</div>
          <div style="margin-top:6px;font-size:13px;color:#666">
            对比：<b>{{ modelA }}</b> vs <b>{{ modelB }}</b>
          </div>
        </div>
      </div>
      <div style="display:flex;gap:8px">
        <div>
          <div style="text-align:center;font-size:12px;margin-bottom:4px">{{ modelA }}</div>
          <svg :ref="el => svgRefs[0] = el"></svg>
        </div>
        <div>
          <div style="text-align:center;font-size:12px;margin-bottom:4px">{{ modelB }}</div>
          <svg :ref="el => svgRefs[1] = el"></svg>
        </div>
        <div>
          <div style="text-align:center;font-size:12px;margin-bottom:4px">差异 (A−B)</div>
          <svg :ref="el => svgRefs[2] = el"></svg>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, watch, computed, nextTick } from 'vue'
import * as d3 from 'd3'
import data from '../data/vl_attention_data.json'

const attentionData = data.attention_data || {};

const props = defineProps({ modelA: String, modelB: String, sampleId: String })
const svgRefs = ref([null, null, null])

const info = computed(() => attentionData[`${props.modelA}__${props.sampleId}`] || {})

function drawHeatmap(svgEl, matrix, colorScale) {
  const rows = matrix.length, cols = matrix[0].length
  const cell = 28
  const labelH = 14
  d3.select(svgEl).selectAll('*').remove()
  const svg = d3.select(svgEl).attr('width', cols * cell).attr('height', rows * cell + labelH)
  const g = svg.append('g').attr('transform', `translate(0,${labelH})`)
  matrix.forEach((row, i) =>
    row.forEach((val, j) => {
      g.append('rect')
        .attr('x', j * cell).attr('y', i * cell)
        .attr('width', cell - 1).attr('height', cell - 1)
        .attr('fill', colorScale(val))
    })
  )
}

function draw() {
  const a = attentionData[`${props.modelA}__${props.sampleId}`]
  const b = attentionData[`${props.modelB}__${props.sampleId}`]
  if (!a || !b) return

  const diff = a.weights.map((row, i) => row.map((v, j) => v - b.weights[i][j]))

  const scaleA = d3.scaleSequential(d3.interpolateBlues).domain([0, 1])
  const scaleB = d3.scaleSequential(d3.interpolateBlues).domain([0, 1])
  const scaleDiff = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1])

  drawHeatmap(svgRefs.value[0], a.weights, scaleA)
  drawHeatmap(svgRefs.value[1], b.weights, scaleB)
  drawHeatmap(svgRefs.value[2], diff, scaleDiff)
}

watch(() => [props.modelA, props.modelB, props.sampleId], async () => {
  await nextTick()
  draw()
})
</script>
