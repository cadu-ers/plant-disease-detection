"""IA generativa (LLM) sobre o diagnóstico: cliente da API, laudo, conversa e avaliação.

O LLM NUNCA decide o diagnóstico: quem decide é o sistema especialista
(ia.expert_system). O LLM só explica o resultado em linguagem simples, converte
respostas em texto livre em sintomas (validados antes de chegar ao motor) e serve
de comparação na avaliação crítica.
"""
