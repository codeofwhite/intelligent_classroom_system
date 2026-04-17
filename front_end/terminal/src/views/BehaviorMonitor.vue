<template>
  <div class="page">
    <h1>>> 学生行为实时监测</h1>

    <div class="section">
      <h3>> 模型配置</h3>
      <div class="row">
        <label>当前模型：</label>
        <select v-model="selectedModel" @change="changeModel">
          <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
        </select>
        <span v-if="switching">加载中...</span>
      </div>
    </div>

    <div class="section">
      <h3>> 实时监控画面</h3>
      <div class="video-container">
        <img src="http://localhost:5000/video_feed" class="stream" />
      </div>
      <p style="margin-top:10px">>> 状态：<span style="color:green">● 正在实时分析</span></p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const modelOptions = ref([])
const selectedModel = ref('')
const switching = ref(false)

async function fetchModels() {
  try {
    const res = await axios.get('http://localhost:5000/get_models')
    modelOptions.value = res.data.models
    selectedModel.value = res.data.current
  } catch (e) {}
}

async function changeModel() {
  switching.value = true
  try {
    await axios.post('http://localhost:5000/switch_model', {
      model_name: selectedModel.value
    })
  } catch (e) {}
  switching.value = false
}

onMounted(() => fetchModels())
</script>

<style scoped>
.page { padding: 30px; }
.section {
  background: #111;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
}
.row { display: flex; gap: 12px; align-items: center; }
select { background:#222; color:#0f0; padding:6px 10px; border:1px solid #444; }
.video-container {
  width: 680px;
  border: 2px solid #0f0;
  background: #000;
}
.stream { width: 100%; display:block; }
</style>