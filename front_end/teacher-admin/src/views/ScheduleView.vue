<template>
  <div class="course-page">
    <h2>📅 我的课程安排</h2>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>星期</th>
            <th>节次</th>
            <th>班级</th>
            <th>课程</th>
            <th>教室</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(item, idx) in list" :key="idx">
            <td>{{ weekMap[item.week_day] }}</td>
            <td>第 {{ item.section }} 节</td>
            <td>{{ item.class_name }}</td>
            <td>{{ item.course_name }}</td>
            <td>{{ item.classroom }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const list = ref([])
const weekMap = ref({ 1: '周一', 2: '周二', 3: '周三', 4: '周四', 5: '周五', 6: '周六', 7: '周日' })

onMounted(() => {
  const user = JSON.parse(localStorage.getItem('userInfo') || '{}')
  const teacher_code = user.teacher_code || 'T2025001'

  fetch('http://localhost:5002/api/teacher/course_schedule', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ teacher_code })
  }).then(res => res.json())
    .then(data => {
      list.value = data.list || []
    })
})
</script>

<style scoped>
.course-page {
  padding: 20px;
}

.table-container {
  background: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 1px 5px #00000010;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

th {
  background: #f5f7fa;
}
</style>