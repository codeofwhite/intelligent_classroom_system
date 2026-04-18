<template>
  <div class="console-page">
    <h1>> TERMINAL AI 监控系统</h1>
    <p>> 系统启动成功 ✅</p>

    <div class="info-card">
      <p>> 教师：{{ teacherName }}</p>
      <p>> 班级：{{ className }}</p>
      <p>> 课程：{{ subject }}</p>
      <p>> 学生总数：{{ studentCount }} 人</p>
    </div>

    <div class="guide">
      <p>> 人脸识别签到：<span>/face-sign</span></p>
      <p>> 课堂实时监测：<span>/class-monitor</span></p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const teacherName = ref('')
const className = ref('')
const subject = ref('')
const studentCount = ref(0)

const loadClassInfo = async () => {
  const user = JSON.parse(localStorage.getItem('terminalUser'))
  if (!user) return

  teacherName.value = user.name

  try {
    const { data } = await axios.post('http://localhost:5001/teacher-class', {
      user_id: user.id
    })
    className.value = data.class_name
    subject.value = data.subject
    studentCount.value = data.student_count
  } catch (err) {
    console.error('加载课室信息失败')
  }
}

onMounted(() => {
  loadClassInfo()
})
</script>

<style scoped>
.console-page {
  padding: 30px;
  line-height: 1.8;
}

.info-card {
  border: 1px solid #0f0;
  padding: 20px;
  margin: 20px 0;
  background: #111;
}

.guide {
  margin-top: 30px;
}
.guide span {
  color: #fff;
  font-weight: bold;
}
</style>