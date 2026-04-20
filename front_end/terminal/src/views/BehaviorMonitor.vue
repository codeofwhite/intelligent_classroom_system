<template>
  <div class="monitor-page">
    <h1>>> 📊 课堂实时监测看板</h1>

    <div class="split-container">
      <!-- 左侧：视频监控 -->
      <div class="left-section">
        <div class="box">
          <h3>> 实时画面</h3>
          <!-- 固定尺寸手机横屏窗口 -->
          <div class="video-phone-frame">
            <img src="http://localhost:5002/video_feed" class="video-feed" />
          </div>
          <p class="status">
            状态：<span class="green">● 实时分析中</span>
          </p>
        </div>

        <div class="box">
          <h3>> 模型配置</h3>
          <div class="row">
            <label>当前模型：</label>
            <select v-model="selectedModel" @change="changeModel">
              <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
            </select>
            <span v-if="switching" class="tip">切换中...</span>
          </div>
        </div>
      </div>

      <!-- 右侧：数据看板 + 日志 -->
      <div class="right-section">
        <div class="box">
          <h3>> 课堂状态</h3>
          <div class="info-grid">
            <div>
              <label>班级</label>
              <span>{{ className }}</span>
            </div>
            <div>
              <label>教师</label>
              <span>{{ teacherName }}</span>
            </div>
            <div>
              <label>应到</label>
              <span>{{ totalStudents }} 人</span>
            </div>
            <div>
              <label>实到</label>
              <span>{{ presentStudents }} 人</span>
            </div>
            <div>
              <label>专注度</label>
              <span class="green">{{ focusRate }}%</span>
            </div>
            <div>
              <label>抬头率</label>
              <span class="green">{{ lookUpRate }}%</span>
            </div>
          </div>
        </div>

        <div class="box">
          <h3>> 行为统计</h3>
          <div class="stat-grid">
            <div>
              <label>🙋‍♂️ 举手</label>
              <span>{{ handUp }}</span>
            </div>
            <div>
              <label>🙆‍♂️ 抬头</label>
              <span>{{ lookUp }}</span>
            </div>
            <div>
              <label>🙅‍♂️ 低头</label>
              <span>{{ lookDown }}</span>
            </div>
            <div>
              <label>🚫 异常</label>
              <span>{{ abnormal }}</span>
            </div>
          </div>
        </div>

        <div class="box log-box">
          <h3>> 实时日志</h3>
          <div class="logs">
            <p v-for="(log, idx) in logs" :key="idx" class="log-item">
              {{ log }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const modelOptions = ref([])
const selectedModel = ref('')
const switching = ref(false)

const className = ref('高一(1)班')
const teacherName = ref('王老师')
const totalStudents = ref(3)
const presentStudents = ref(3)
const focusRate = ref(89)
const lookUpRate = ref(94)

const handUp = ref(0)
const lookUp = ref(2)
const lookDown = ref(1)
const abnormal = ref(0)

const logs = ref([
  '[20:23:10] 系统启动成功',
  '[20:23:12] 行为模型加载完成',
  '[20:23:15] 张三 · 抬头',
  '[20:23:18] 李四 · 低头',
  '[20:23:20] 王五 · 抬头'
])

async function fetchModels() {
  try {
    const res = await axios.get('http://localhost:5002/get_models')
    modelOptions.value = res.data.models
    selectedModel.value = res.data.current
  } catch (e) {}
}

async function changeModel() {
  switching.value = true
  try {
    await axios.post('http://localhost:5002/switch_model', {
      model_name: selectedModel.value
    })
  } catch (e) {}
  switching.value = false
}

onMounted(() => {
  fetchModels()
})
</script>

<style scoped>
.monitor-page {
  padding: 24px;
  color: #0f0;
  background: #000;
  min-height: 100vh;
}

.split-container {
  display: flex;
  gap: 20px;
  margin-top: 16px;
}

/* 左侧固定宽度，避免乱跑 */
.left-section {
  width: 452px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.right-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.box {
  background: #111;
  border: 1px solid #0f0;
  padding: 16px;
  border-radius: 6px;
}

/* ========== 核心：手机横屏固定尺寸 ========== */
.video-phone-frame {
  /* 手机横屏比例 16:9 */
  width: 420px;
  height: 236px;
  border: 2px solid #0f0;
  background: #000;
  overflow: hidden;
  position: relative;
}

.video-feed {
  position: absolute;
  width: 100%;
  height: 100%;
  object-fit: cover; /* 自动裁剪，不变形 */
}

.status {
  margin-top: 8px;
  font-size: 14px;
}

.green {
  color: lightgreen;
}

.row {
  display: flex;
  align-items: center;
  gap: 12px;
}

select {
  background: #222;
  color: #0f0;
  border: 1px solid #444;
  padding: 6px;
}

.info-grid,
.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.info-grid > div,
.stat-grid > div {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
}

.log-box {
  min-height: 240px;
}

.logs {
  height: 200px;
  overflow-y: auto;
  background: #000;
  padding: 10px;
  border: 1px solid #333;
}

.log-item {
  margin: 4px 0;
  font-size: 14px;
}
</style>