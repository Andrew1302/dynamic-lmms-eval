#!/bin/bash
# =============================================================================
# 00_wsl_fix_key.sh — Copiar chave SSH do Windows para o WSL com permissões corretas
# =============================================================================
#
# No WSL, chaves em /mnt/c/... ficam com permissão 0777 porque o sistema de
# arquivos do Windows não suporta chmod por padrão. O SSH recusa usar chaves
# com permissões abertas, mostrando o erro:
#
#   WARNING: UNPROTECTED PRIVATE KEY FILE!
#   Permissions 0777 for '/mnt/c/Users/.../.ssh/vm_key' are too open.
#
# Este script copia a chave para ~/.ssh/ dentro do WSL, onde chmod funciona
# normalmente, e exporta SSH_KEY apontando para o novo caminho.
#
# USO
#     source ./00_wsl_fix_key.sh /mnt/c/Users/SeuUsuario/.ssh/sua_chave
#
#     Usar "source" (ou ".") é importante para que o export de SSH_KEY
#     persista na sessão atual do terminal. Se rodar com ./00_wsl_fix_key.sh,
#     a variável só existirá dentro do subshell do script.
#
# =============================================================================

if [ $# -eq 0 ]; then
    echo "USO: source ./00_wsl_fix_key.sh /mnt/c/Users/SeuUsuario/.ssh/sua_chave"
    echo ""
    echo "Copia a chave SSH do Windows para ~/.ssh/ no WSL com chmod 600"
    echo "e exporta SSH_KEY apontando para o novo caminho."
    return 1 2>/dev/null || exit 1
fi

WINDOWS_KEY="$1"

if [ ! -f "$WINDOWS_KEY" ]; then
    echo "ERRO: Arquivo não encontrado: $WINDOWS_KEY"
    return 1 2>/dev/null || exit 1
fi

# Garantir que ~/.ssh existe com permissões corretas
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Copiar a chave
KEY_NAME=$(basename "$WINDOWS_KEY")
WSL_KEY="$HOME/.ssh/$KEY_NAME"

cp "$WINDOWS_KEY" "$WSL_KEY"
chmod 600 "$WSL_KEY"

# Exportar SSH_KEY
export SSH_KEY="$WSL_KEY"

echo "✅ Chave copiada: $WINDOWS_KEY -> $WSL_KEY"
echo "   Permissões: $(stat -c '%a' "$WSL_KEY")"
echo "   SSH_KEY exportada: $SSH_KEY"
echo ""
echo "Agora pode rodar os scripts normalmente."
