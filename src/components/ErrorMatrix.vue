<template>
  <div style="padding:16px">
    <div style="margin-bottom:12px;display:flex;gap:12px;align-items:center">
      <label style="font-size:13px">
        <input type="checkbox" v-model="showOnlyErrors" @change="redraw" />
        只显示有模型答错的样本
      </label>
      <span style="font-size:12px;color:#999">（{{ filteredSamples.length }} / {{ allSamples.length }} 个样本）</span>
    </div>
    <div style="overflow-x:auto;max-height:70vh">
      <svg ref="svgRef"></svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import * as d3 from 'd3'

const props = defineProps({ data: Array })
const emit = defineEmits(['cell-click'])
const svgRef = ref(null)
const showOnlyErrors = ref(true)

const models = computed(() => [...new Set(props.data.map(d => d.model))])
const allSamples = computed(() => [...new Set(props.data.map(d => d.sample_id))])

const filteredSamples = computed(() => {
  if (!showOnlyErrors.value) return allSamples.value
  // 只保留至少一个模型答错的样本
  const errorSamples = new Set()
  props.data.forEach(d => { if (!d.correct) errorSamples.add(d.sample_id) })
  return allSamples.value.filter(s => errorSamples.has(s))
})

function redraw() {
  const samples = filteredSamples.value
  const margin = { top: 30, right: 20, bottom: 60, left: 80 }
  const cellSize = 40
  const w = samples.length * cellSize + margin.left + margin.right
  const h = models.value.length * cellSize + margin.top + margin.bottom

  const svg = d3.select(svgRef.value)
    .attr('width', w).attr('height', h)
  svg.selectAll('*').remove()

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

  const x = d3.scaleBand().domain(samples).range([0, samples.length * cellSize]).padding(0.05)
  const y = d3.scaleBand().domain(models.value).range([0, models.value.length * cellSize]).padding(0.05)

  // axes
  g.append('g').attr('transform', `translate(0,${models.value.length * cellSize})`)
    .call(d3.axisBottom(x))
    .selectAll('text').attr('transform', 'rotate(-40)').style('text-anchor', 'end').style('font-size', '10px')

  g.append('g').call(d3.axisLeft(y)).selectAll('text').style('font-size', '13px')

  // tooltip
  const tooltip = d3.select('body').selectAll('.matrix-tooltip').data([0]).join('div')
    .attr('class', 'matrix-tooltip')
    .style('position', 'absolute')
    .style('background', 'rgba(0,0,0,0.8)')
    .style('color', '#fff')
    .style('padding', '6px 10px')
    .style('border-radius', '4px')
    .style('font-size', '12px')
    .style('pointer-events', 'none')
    .style('opacity', 0)

  // cells
  const filtered = props.data.filter(d => samples.includes(d.sample_id))
  g.selectAll('rect').data(filtered).join('rect')
    .attr('x', d => x(d.sample_id))
    .attr('y', d => y(d.model))
    .attr('width', x.bandwidth())
    .attr('height', y.bandwidth())
    .attr('fill', d => d.correct ? '#4caf50' : '#f44336')
    .attr('opacity', 0.85)
    .attr('rx', 3)
    .attr('stroke', d => d.correct ? null : '#000')
    .attr('stroke-width', d => d.correct ? 0 : 1.5)
    .style('cursor', 'pointer')
    .on('mouseover', function(e, d) {
      d3.select(this).attr('stroke', '#333').attr('stroke-width', 3)
      tooltip.style('opacity', 1)
        .html(`<b>${d.model}</b><br/>样本 ${d.sample_id}<br/>${d.correct ? '✓ 正确' : '✗ 错误'}`)
        .style('left', (e.pageX + 10) + 'px')
        .style('top', (e.pageY - 20) + 'px')
    })
    .on('mouseout', function(e, d) {
      d3.select(this).attr('stroke', d.correct ? null : '#000').attr('stroke-width', d.correct ? 0 : 1.5)
      tooltip.style('opacity', 0)
    })
    .on('click', (_, d) => emit('cell-click', { model: d.model, sample_id: d.sample_id }))
}

onMounted(redraw)
watch(() => props.data, redraw)
watch(showOnlyErrors, redraw)
</script>
