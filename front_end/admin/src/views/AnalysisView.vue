<template>
  <div class="report-page">
    <h2>📊 课堂深度行为分析报告</h2>

    <!-- 1. 概况卡片 -->
    <div class="card info-card">
      <div>班级：{{ data.class_name }}</div>
      <div>课程：{{ data.course_name }}</div>
      <div>专注率：{{ data.focus_rate }}%</div>
      <div>分心率：{{ data.distract_rate }}%</div>
    </div>

    <!-- 2. 行为饼图 -->
    <div class="card">
      <h3>行为分布</h3>
      <pie-chart :data="data.behavior_counts" />
    </div>

    <!-- 3. 专注度趋势图 -->
    <div class="card">
      <h3>课堂专注度变化</h3>
      <line-chart :data="data.time_trend" />
    </div>

    <!-- 4. 分心学生排行 -->
    <div class="card">
      <h3>分心次数学生排行</h3>
      <div v-for="item in data.student_ranks">
        学生{{ item.student_id }}：{{ item.distract_count }}次
      </div>
    </div>

    <!-- 5. 关键帧 -->
    <div class="card">
      <h3>课堂异常关键帧</h3>
      <img v-for="url in data.key_frames" :src="url" width="100%">
    </div>

    <!-- 6. AI 总结 -->
    <div class="card">
      <h3>🤖 AI 课堂总结与家校建议</h3>
      <div>{{ data.ai_summary }}</div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      data: {}
    }
  },
  mounted() {
    // 调用后端接口
    fetch("/api/class_report/123").then(res => res.json()).then(data => {
      this.data = data
    })
  }
}
</script>

<style>
.report-page {
  max-width: 1200px;
  margin: auto;
  padding: 20px;
}
.card {
  background: white;
  padding: 20px;
  border-radius: 16px;
  margin-bottom: 20px;
}
</style>