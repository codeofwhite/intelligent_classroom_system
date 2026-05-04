<template>
  <div class="page">
    <h1>>> 人脸识别签到系统</h1>

    <!--  Flask 视频流  -->
    <div class="video-box">
      <img src="http://localhost:5003/video_feed" class="live-video" />
    </div>

    <div class="logs">
      <h3>> 签到记录</h3>
      <div v-for="(log, i) in logs" :key="i">{{ log }}</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const logs = ref([
  '> 系统启动成功',
  '> 连接人脸识别服务...',
  '> 实时签到已启动'
])

// 定时拉取签到日志
async function fetchLogs() {
  try {
    const res = await axios.get('http://localhost:5003/get_sign_log')
    logs.value = res.data.logs
  } catch (err) {
    console.log('获取日志失败')
  }
}

onMounted(() => {
  fetchLogs()
  setInterval(fetchLogs, 2000) // 2秒刷新一次日志
})
</script>

<style scoped>
.page {
  padding: 30px;
  background: #000;
  color: #0f0;
  font-family: "Courier New", monospace;
  min-height: 100vh;
}

.video-box {
  width: 640px;
  height: 480px;
  background: #111;
  border: 2px solid #0f0;
  position: relative;
  margin: 20px 0;
}

.live-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.logs {
  margin-top: 20px;
  background: #111;
  padding: 15px;
  border: 1px solid #333;
  max-height: 250px;
  overflow-y: auto;
}
</style>