<template>
  <div style="display:grid;grid-template-columns:1fr 1.5fr;height:100vh;font-family:sans-serif">
    <div style="border-right:1px solid #eee;overflow:auto">
      <div style="padding:16px 16px 0;font-weight:bold;font-size:15px">模型-样本错误矩阵</div>
      <div style="padding:8px 16px;font-size:12px;color:#888">
        绿色=正确 红色=错误，透明度=置信度。点击格子查看注意力对比。
      </div>
      <ErrorMatrix :data="matrixData" @cell-click="onCellClick" />
    </div>
    <div style="overflow:auto">
      <div style="padding:16px 16px 0;font-weight:bold;font-size:15px">注意力差异热力图</div>
      <div style="padding:4px 16px;font-size:12px;color:#888;display:flex;align-items:center;gap:8px">
        对比基准：
        <select v-model="selected.modelB" style="font-size:12px">
          <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
      <AttentionHeatmap
        :modelA="selected.modelA"
        :modelB="selected.modelB"
        :sampleId="selected.sampleId"
      />
    </div>
  </div>
</template>

<script setup>
import { reactive, computed } from 'vue'
import ErrorMatrix from './components/ErrorMatrix.vue'
import AttentionHeatmap from './components/AttentionHeatmap.vue'
import data from './data/vl_attention_data.json'

const matrixData = data.matrix_data || [];
const models = [...new Set(matrixData.map(d => d.model))]
const selected = reactive({ modelA: models[0], modelB: models[1], sampleId: null })

function onCellClick({ model, sample_id }) {
  selected.modelA = model
  selected.sampleId = sample_id
}
</script>
