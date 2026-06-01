<template>
  <div class="report-page">
    <h2>📝 学生课堂行为报告</h2>

    <!-- 顶部：班级选择 + 周标签 -->
    <div class="top-bar">
      <div class="bar-left">
        <label>班级：</label>
        <select v-model="classId" @change="onClassChange">
          <option value="">-- 选择班级 --</option>
          <option v-for="c in classList" :key="c.class_code" :value="c.class_code">
            {{ c.class_name }}
          </option>
        </select>
      </div>
    </div>

    <!-- 周标签页 -->
    <div class="week-tabs" v-if="classId && weekTabs.length > 0">
      <button
        v-for="tab in weekTabs"
        :key="tab.key"
        class="week-tab"
        :class="{ active: currentWeek === tab.key }"
        @click="currentWeek = tab.key"
      >
        {{ tab.label }}
        <span class="tab-count">{{ tab.count }}</span>
      </button>
    </div>

    <!-- 双栏布局 -->
    <div class="report-layout" v-if="classId">
      <!-- 左侧：会话时间线 -->
      <div class="session-sidebar">
        <div class="sidebar-header">
          <span>📅 课堂记录</span>
          <span class="sidebar-count">{{ filteredSessions.length }} 节</span>
        </div>
        <div class="session-timeline">
          <div
            v-for="session in filteredSessions"
            :key="session.key"
            class="timeline-item"
            :class="{ active: activeSession?.key === session.key }"
            @click="selectSession(session)"
          >
            <div class="timeline-dot" :class="{ done: session.unscoredCount === 0 }"></div>
            <div class="timeline-content">
              <div class="timeline-date">{{ session.shortLabel }}</div>
              <div class="timeline-meta">
                <span class="meta-section">{{ session.lesson_section || '课堂' }}</span>
                <span class="meta-status" :class="session.unscoredCount === 0 ? 'all-done' : 'has-pending'">
                  {{ session.unscoredCount === 0 ? '✓ 已完成' : session.unscoredCount + '人待评' }}
                </span>
              </div>
            </div>
          </div>
        </div>
        <div v-if="filteredSessions.length === 0" class="sidebar-empty">暂无课堂记录</div>
      </div>

      <!-- 右侧：选中课堂的学生详情 -->
      <div class="session-detail" v-if="activeSession">
        <div class="detail-header">
          <div class="detail-title">
            <h3>{{ activeSession.label }}</h3>
            <span class="detail-section">{{ activeSession.lesson_section }}</span>
          </div>
          <div class="detail-stats">
            <span class="stat done">{{ activeSession.scoredCount }} 已评</span>
            <span class="stat pending">{{ activeSession.unscoredCount }} 未评</span>
          </div>
        </div>

        <div class="student-search">
          <input v-model="activeSession.search" placeholder="🔍 搜索学生姓名..." />
        </div>

        <div class="section-group" v-if="filterStudents(activeSession, false).length > 0">
          <div class="section-label pending-label">⏳ 未评（{{ filterStudents(activeSession, false).length }}人）</div>
          <div class="student-grid">
            <div v-for="stu in filterStudents(activeSession, false)" :key="stu.student_code"
              class="student-chip pending" @click="openReport(activeSession, stu)">
              <span class="chip-name">{{ stu.name }}</span>
              <span class="chip-focus">{{ stu.focus_rate }}%</span>
            </div>
          </div>
        </div>

        <div class="section-group" v-if="filterStudents(activeSession, true).length > 0">
          <div class="section-label done-label">✅ 已评（{{ filterStudents(activeSession, true).length }}人）</div>
          <div class="student-grid">
            <div v-for="stu in filterStudents(activeSession, true)" :key="stu.student_code"
              class="student-chip done" @click="openReport(activeSession, stu)">
              <span class="chip-name">{{ stu.name }}</span>
              <span class="chip-score">{{ stu.teacher_score }}分</span>
            </div>
          </div>
        </div>

        <div v-if="filterStudents(activeSession, false).length === 0 && filterStudents(activeSession, true).length === 0" class="detail-empty">
          无匹配学生
        </div>
      </div>

      <div class="session-detail empty-state" v-else-if="classId && sessionList.length > 0">
        <div class="empty-icon">👈</div>
        <p>请从左侧选择一节课堂</p>
      </div>

      <div class="session-detail empty-state" v-else-if="classId">
        <div class="empty-icon">📭</div>
        <p>该班级暂无课堂记录</p>
      </div>
    </div>

    <div v-if="!classId" class="empty big">请先选择班级查看课堂报告</div>

    <!-- 报告编辑弹窗 -->
    <div class="modal" v-if="editingStudent" @click="closeReport">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <div>
            <h3>{{ editingStudent.name }} 的课堂报告</h3>
            <p class="modal-sub">{{ editingSession?.label }} · {{ editingSession?.lesson_section }}</p>
          </div>
          <button @click="closeReport">×</button>
        </div>
        <div class="modal-body">
          <div class="behavior-card">
            <h4>📊 行为统计</h4>
            <div class="behavior-grid" v-if="editingStudent.behaviors_json">
              <div v-for="(cnt, label) in parseBehaviors(editingStudent.behaviors_json)" :key="label" class="beh-item">
                <span class="beh-label">{{ label }}</span>
                <span class="beh-val">{{ cnt }}</span>
              </div>
            </div>
            <div class="behavior-grid" v-else>
              <div class="beh-item"><span class="beh-label">正常坐姿</span><span class="beh-val">{{ editingStudent.normal_posture }}</span></div>
              <div class="beh-item"><span class="beh-label">举手</span><span class="beh-val">{{ editingStudent.raised_hand }}</span></div>
              <div class="beh-item"><span class="beh-label">低头</span><span class="beh-val">{{ editingStudent.looking_down }}</span></div>
            </div>
            <div class="focus-row">
              <span>专注度</span>
              <span class="focus-val" :class="focusClass(editingStudent.focus_rate)">{{ editingStudent.focus_rate }}%</span>
            </div>
          </div>
          <div class="edit-area">
            <label>💡 AI 自动评语</label>
            <textarea v-model="editForm.aiComment" rows="3" placeholder="点击 AI 分析自动生成"></textarea>
            <div class="score-row">
              <div class="score-field">
                <label>📝 老师评分（1-100）</label>
                <input v-model="editForm.score" type="number" min="1" max="100" />
              </div>
            </div>
            <label>👨‍🏫 老师评语</label>
            <textarea v-model="editForm.teacherComment" rows="2" placeholder="输入老师评语"></textarea>
            <div class="modal-btns">
              <button class="btn-ai" @click="runAI" :disabled="aiLoading">{{ aiLoading ? '分析中...' : '🤖 AI 自动分析' }}</button>
              <button class="btn-save" @click="saveReport">💾 保存报告</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import axios from 'axios'

const classList = ref([])
const classId = ref('')
const sessionList = ref([])
const studentMap = ref({})
const currentWeek = ref('all')
const activeSession = ref(null)

const editingStudent = ref(null)
const editingSession = ref(null)
const editForm = reactive({ aiComment: '', score: '', teacherComment: '' })
const aiLoading = ref(false)

const loadClasses = async () => {
  try {
    const { data } = await axios.get('http://localhost:5002/api/class/list')
    classList.value = data.list || []
  } catch (err) { console.error(err) }
}

const onClassChange = async () => {
  sessionList.value = []
  activeSession.value = null
  currentWeek.value = 'all'
  if (!classId.value) return
  try {
    const stuRes = await axios.get('http://localhost:5002/api/class/students', { params: { class_code: classId.value } })
    const students = stuRes.data?.students || []
    const stuMap = {}
    students.forEach(s => { stuMap[s.student_code] = s.name })
    studentMap.value = stuMap

    const reportRes = await axios.get('http://localhost:5002/api/report/list_by_class', { params: { class_code: classId.value } })
    const reports = reportRes.data.list || []

    const groups = {}
    reports.forEach(r => {
      const key = r.lesson_time
      if (!groups[key]) {
        const d = new Date(r.lesson_time)
        const week = '日一二三四五六'[d.getDay()]
        groups[key] = {
          key, lesson_time: r.lesson_time, lesson_section: r.lesson_section || '',
          label: `${d.getFullYear()}年${d.getMonth()+1}月${d.getDate()}日 周${week} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`,
          shortLabel: `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`,
          reports: {}, search: '', open: false
        }
      }
      groups[key].reports[r.student_code] = { ...r, name: stuMap[r.student_code] || r.student_code }
    })

    const sessions = Object.values(groups).sort((a, b) => new Date(b.lesson_time) - new Date(a.lesson_time))
    sessions.forEach(s => {
      const all = Object.values(s.reports)
      s.scoredCount = all.filter(r => r.teacher_score != null).length
      s.unscoredCount = all.filter(r => r.teacher_score == null).length
    })
    sessionList.value = sessions
    if (sessions.length > 0) activeSession.value = sessions[0]
  } catch (err) { console.error(err) }
}

const weekTabs = computed(() => {
  const tabs = [{ key: 'all', label: '全部', count: sessionList.value.length }]
  const weekMap = {}
  sessionList.value.forEach(s => {
    const d = new Date(s.lesson_time)
    const start = new Date(d); start.setDate(d.getDate() - d.getDay() + 1)
    const key = start.toISOString().slice(0, 10)
    if (!weekMap[key]) {
      const end = new Date(start); end.setDate(start.getDate() + 6)
      weekMap[key] = { key, label: `${start.getMonth()+1}/${start.getDate()}-${end.getMonth()+1}/${end.getDate()}`, count: 0 }
    }
    weekMap[key].count++
  })
  return [...tabs, ...Object.values(weekMap).sort((a, b) => b.key.localeCompare(a.key))]
})

const filteredSessions = computed(() => {
  if (currentWeek.value === 'all') return sessionList.value
  return sessionList.value.filter(s => {
    const d = new Date(s.lesson_time)
    const start = new Date(d); start.setDate(d.getDate() - d.getDay() + 1)
    return start.toISOString().slice(0, 10) === currentWeek.value
  })
})

const selectSession = (session) => { activeSession.value = session }

const filterStudents = (session, scored) => {
  if (!session) return []
  const all = Object.values(session.reports)
  const keyword = (session.search || '').trim().toLowerCase()
  return all.filter(r => {
    const matchScore = scored ? r.teacher_score != null : r.teacher_score == null
    const matchSearch = !keyword || (r.name && r.name.toLowerCase().includes(keyword))
    return matchScore && matchSearch
  })
}

const parseBehaviors = (json) => { try { return JSON.parse(json) } catch { return {} } }
const focusClass = (rate) => rate >= 85 ? 'good' : rate >= 70 ? 'medium' : 'bad'

const openReport = (session, stu) => {
  editingStudent.value = stu
  editingSession.value = session
  editForm.aiComment = stu.ai_comment || ''
  editForm.score = stu.teacher_score || ''
  editForm.teacherComment = stu.teacher_comment || ''
}
const closeReport = () => { editingStudent.value = null; editingSession.value = null }

const runAI = async () => {
  if (!editingStudent.value) return
  aiLoading.value = true
  try {
    const { data } = await axios.post('http://localhost:5002/api/ai/analyze', {
      student_code: editingStudent.value.student_code,
      normal_posture: editingStudent.value.normal_posture || 0,
      raised_hand: editingStudent.value.raised_hand || 0,
      looking_down: editingStudent.value.looking_down || 0,
      focus_rate: editingStudent.value.focus_rate || 0
    })
    editForm.aiComment = data.comment || ''
  } catch { alert('AI 分析失败') } finally { aiLoading.value = false }
}

const saveReport = async () => {
  if (!editingStudent.value) return
  try {
    await axios.post('http://localhost:5002/api/report/save', {
      id: editingStudent.value.id, student_code: editingStudent.value.student_code,
      class_code: classId.value, lesson_time: editingStudent.value.lesson_time,
      normal_posture: editingStudent.value.normal_posture || 0,
      raised_hand: editingStudent.value.raised_hand || 0,
      looking_down: editingStudent.value.looking_down || 0,
      focus_rate: editingStudent.value.focus_rate || 0,
      ai_comment: editForm.aiComment, teacher_score: editForm.score || null,
      teacher_comment: editForm.teacherComment
    })
    editingStudent.value.teacher_score = editForm.score ? Number(editForm.score) : null
    editingStudent.value.teacher_comment = editForm.teacherComment
    editingStudent.value.ai_comment = editForm.aiComment
    if (editingSession.value) {
      const all = Object.values(editingSession.value.reports)
      editingSession.value.scoredCount = all.filter(r => r.teacher_score != null).length
      editingSession.value.unscoredCount = all.filter(r => r.teacher_score == null).length
    }
    alert('✅ 保存成功！')
    closeReport()
  } catch { alert('保存失败') }
}

onMounted(() => { loadClasses() })
</script>

<style scoped>
.report-page { padding: 24px; background: #f5f7fa; min-height: 100vh; }
h2 { margin: 0 0 20px 0; font-size: 22px; color: #2c3e50; }

.top-bar {
  display: flex; justify-content: space-between; align-items: center;
  background: #e6f7ff; padding: 12px 16px; border-radius: 10px; margin-bottom: 16px;
}
.bar-left { display: flex; align-items: center; gap: 8px; }
.bar-left label { font-weight: 500; color: #333; }
.bar-left select { padding: 6px 12px; border-radius: 6px; border: 1px solid #ccc; }

/* 周标签 */
.week-tabs {
  display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap;
}
.week-tab {
  padding: 6px 14px; border: 1px solid #e0e0e0; border-radius: 16px;
  background: #fff; font-size: 12px; cursor: pointer; color: #666; transition: all 0.2s;
}
.week-tab.active { background: #1890ff; color: #fff; border-color: #1890ff; }
.tab-count { margin-left: 4px; opacity: 0.7; }

/* 双栏布局 */
.report-layout {
  display: flex; gap: 16px; min-height: 60vh;
}

/* 左侧时间线 */
.session-sidebar {
  width: 260px; flex-shrink: 0; background: #fff; border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04); display: flex; flex-direction: column;
}
.sidebar-header {
  padding: 14px 16px; border-bottom: 1px solid #f0f0f0;
  display: flex; justify-content: space-between; align-items: center;
  font-weight: 600; font-size: 14px; color: #2c3e50;
}
.sidebar-count { font-size: 12px; color: #999; font-weight: normal; }

.session-timeline { flex: 1; overflow-y: auto; padding: 8px 0; }

.timeline-item {
  display: flex; align-items: flex-start; gap: 10px; padding: 10px 16px;
  cursor: pointer; transition: background 0.2s; border-left: 3px solid transparent;
}
.timeline-item:hover { background: #f8f9fa; }
.timeline-item.active { background: #f0f5ff; border-left-color: #1890ff; }

.timeline-dot {
  width: 10px; height: 10px; border-radius: 50%; background: #fa8c16;
  margin-top: 4px; flex-shrink: 0;
}
.timeline-dot.done { background: #52c41a; }

.timeline-content { flex: 1; min-width: 0; }
.timeline-date { font-size: 13px; font-weight: 500; color: #2c3e50; margin-bottom: 2px; }
.timeline-meta { display: flex; gap: 8px; align-items: center; }
.meta-section { font-size: 11px; color: #999; background: #f0f0f0; padding: 1px 6px; border-radius: 3px; }
.meta-status { font-size: 11px; }
.meta-status.all-done { color: #52c41a; }
.meta-status.has-pending { color: #fa8c16; }

.sidebar-empty { padding: 30px; text-align: center; color: #ccc; font-size: 13px; }

/* 右侧详情 */
.session-detail {
  flex: 1; background: #fff; border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04); padding: 20px;
  display: flex; flex-direction: column;
}

.detail-header {
  display: flex; justify-content: space-between; align-items: flex-start;
  margin-bottom: 16px; padding-bottom: 14px; border-bottom: 1px solid #f0f0f0;
}
.detail-title h3 { margin: 0; font-size: 16px; color: #2c3e50; }
.detail-section { font-size: 12px; color: #999; }
.detail-stats { display: flex; gap: 8px; }
.stat { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.stat.done { background: #e6ffed; color: #00b42a; }
.stat.pending { background: #fff7e6; color: #fa8c16; }

.student-search { margin-bottom: 12px; }
.student-search input {
  width: 100%; padding: 8px 12px; border: 1px solid #e0e0e0;
  border-radius: 8px; font-size: 13px; outline: none; box-sizing: border-box;
}
.student-search input:focus { border-color: #1890ff; }

.section-group { margin-bottom: 12px; }
.section-label { font-size: 13px; font-weight: 600; margin: 8px 0 8px 4px; }
.pending-label { color: #fa8c16; }
.done-label { color: #00b42a; }

.student-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.student-chip {
  display: flex; align-items: center; gap: 6px; padding: 7px 12px;
  border-radius: 18px; cursor: pointer; font-size: 12px;
  transition: all 0.2s; border: 1px solid transparent;
}
.student-chip:hover { transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.student-chip.pending { background: #fffbe6; border-color: #ffe58f; color: #ad6800; }
.student-chip.done { background: #f6ffed; border-color: #b7eb8f; color: #389e0d; }
.chip-name { font-weight: 500; }
.chip-focus, .chip-score { font-size: 11px; opacity: 0.85; }

.detail-empty { text-align: center; padding: 40px; color: #ccc; }

.empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.empty-icon { font-size: 36px; margin-bottom: 10px; }
.empty-state p { color: #999; font-size: 14px; }
.empty { text-align: center; padding: 40px; color: #999; font-size: 14px; }
.empty.big { font-size: 16px; padding: 80px; }

/* 弹窗 */
.modal {
  position: fixed; inset: 0; background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center;
  z-index: 999; backdrop-filter: blur(3px);
}
.modal-content {
  background: #fff; width: 90%; max-width: 520px; border-radius: 18px;
  max-height: 90vh; overflow-y: auto; box-shadow: 0 20px 40px rgba(0,0,0,0.2); animation: modalUp 0.3s ease;
}
@keyframes modalUp { from { transform: translateY(30px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
.modal-header {
  padding: 20px; border-bottom: 1px solid #f0f0f0;
  display: flex; justify-content: space-between; align-items: flex-start;
}
.modal-header h3 { margin: 0; font-size: 18px; color: #2c3e50; }
.modal-sub { margin: 4px 0 0 0; font-size: 13px; color: #999; }
.modal-header button {
  background: #f5f5f5; border: none; width: 30px; height: 30px;
  border-radius: 50%; font-size: 16px; cursor: pointer; color: #999;
}
.modal-body { padding: 20px; }

.behavior-card { background: #f8f9fd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
.behavior-card h4 { margin: 0 0 12px 0; font-size: 14px; color: #555; }
.behavior-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 12px; }
.beh-item { text-align: center; background: #fff; padding: 8px; border-radius: 8px; }
.beh-label { display: block; font-size: 11px; color: #999; margin-bottom: 4px; }
.beh-val { font-size: 16px; font-weight: 600; color: #2c3e50; }
.focus-row {
  display: flex; justify-content: space-between; align-items: center;
  padding-top: 10px; border-top: 1px solid #eee; font-size: 14px; color: #555;
}
.focus-val { font-size: 22px; font-weight: 700; }
.focus-val.good { color: #00b42a; }
.focus-val.medium { color: #fa8c16; }
.focus-val.bad { color: #ff4d4f; }

.edit-area { display: flex; flex-direction: column; gap: 10px; }
.edit-area label { font-size: 13px; font-weight: 500; color: #555; }
.edit-area textarea {
  padding: 10px; border: 1px solid #e0e0e0; border-radius: 8px;
  font-size: 13px; resize: none; font-family: inherit;
}
.score-row { display: flex; gap: 12px; }
.score-field { flex: 1; }
.score-field input {
  width: 100%; padding: 8px 12px; border: 1px solid #e0e0e0;
  border-radius: 8px; font-size: 14px; box-sizing: border-box;
}
.modal-btns { display: flex; gap: 10px; margin-top: 6px; }
.btn-ai, .btn-save {
  flex: 1; padding: 10px; border: none; border-radius: 8px;
  font-size: 14px; cursor: pointer; font-weight: 500;
}
.btn-ai { background: #722ed1; color: #fff; }
.btn-ai:disabled { opacity: 0.6; cursor: not-allowed; }
.btn-save { background: #1890ff; color: #fff; }
</style>