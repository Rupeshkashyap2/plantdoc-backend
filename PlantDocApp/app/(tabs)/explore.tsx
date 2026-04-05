import { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ScrollView, ActivityIndicator, KeyboardAvoidingView, Platform } from 'react-native';

export default function AIAssistant() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hello! I am your plant health assistant. Ask me anything about plant diseases, treatments, or care tips!' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userInput = input;
    const userMsg = { role: 'user', text: userInput };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const response = await fetch('https://plantdoc-backend-5vll.onrender.com/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput })
      });
      const data = await response.json();
      const aiMsg = { role: 'assistant', text: data.reply };
      setMessages(prev => [...prev, aiMsg]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Connection error: ' + e.message }]);
    }
    setLoading(false);
  };

  return (
    <KeyboardAvoidingView style={styles.container} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
      <View style={styles.header}>
        <Text style={styles.headerText}>🤖 AI Assistant</Text>
        <Text style={styles.headerSub}>Ask anything about plants</Text>
      </View>
      <ScrollView style={styles.chat}>
        {messages.map((msg, i) => (
          <View key={i} style={[styles.bubble, msg.role === 'user' ? styles.userBubble : styles.aiBubble]}>
            <Text style={[styles.bubbleText, msg.role === 'user' ? styles.userText : styles.aiText]}>{msg.text}</Text>
          </View>
        ))}
        {loading && <ActivityIndicator color="#4ade80" style={{marginTop: 10}}/>}
      </ScrollView>
      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask about plant diseases..."
          placeholderTextColor="#6b7280"
          onSubmitEditing={sendMessage}
        />
        <TouchableOpacity style={styles.sendBtn} onPress={sendMessage}>
          <Text style={styles.sendText}>➤</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0d1117' },
  header: { backgroundColor: '#0a2a1a', padding: 40, alignItems: 'center' },
  headerText: { color: '#4ade80', fontSize: 24, fontWeight: 'bold' },
  headerSub: { color: '#6b7280', fontSize: 13, marginTop: 4 },
  chat: { flex: 1, padding: 16 },
  bubble: { borderRadius: 16, padding: 12, marginBottom: 10, maxWidth: '80%' },
  aiBubble: { backgroundColor: '#1a2a1f', alignSelf: 'flex-start' },
  userBubble: { backgroundColor: '#4ade80', alignSelf: 'flex-end' },
  aiText: { color: '#d1d5db', fontSize: 14, lineHeight: 20 },
  userText: { color: '#0a2a1a', fontSize: 14, fontWeight: '600' },
  inputRow: { flexDirection: 'row', padding: 12, gap: 8, backgroundColor: '#111827' },
  input: {
    flex: 1, backgroundColor: '#1f2937', borderRadius: 20,
    padding: 12, color: '#e5e7eb', fontSize: 14,
    borderWidth: 0.5, borderColor: '#374151'
  },
  sendBtn: {
    width: 44, height: 44, borderRadius: 22,
    backgroundColor: '#4ade80', alignItems: 'center', justifyContent: 'center'
  },
  sendText: { color: '#0a2a1a', fontSize: 18, fontWeight: 'bold' },
});