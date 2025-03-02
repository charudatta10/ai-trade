import ai_agent_stock_trading_bot

import uvicorn
    
    
uvicorn.run(ai_agent_stock_trading_bot.app, host="localhost", port=8000)

