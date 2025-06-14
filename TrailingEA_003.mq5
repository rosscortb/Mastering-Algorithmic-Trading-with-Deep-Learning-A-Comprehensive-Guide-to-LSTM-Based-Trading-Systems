//+------------------------------------------------------------------+
//|                                             TrailingStopEA.mq5 |
//|                                        Desarrollado por Claude |
//|                                                               |
//+------------------------------------------------------------------+
#property copyright "Claude"
#property link      ""
#property version   "1.10"
#property strict

// Incluimos las bibliotecas necesarias
#include <Trade\Trade.mqh>

// Definición de constantes y variables globales
enum TRAILING_MODE
{
   ALL_ORDERS = 0,        // Todas las órdenes
   BY_MAGIC_NUMBER = 1,   // Filtrar por Magic Number
   BY_SL_TP_LEVELS = 2    // Filtrar por niveles de SL/TP
};

// Parámetros generales del trailing stop
input TRAILING_MODE  TrailingMode = ALL_ORDERS;   // Modo de operación del trailing
input double         TrailingStopPoints = 20;    // Distancia del Trailing Stop en puntos
input double         TrailingStepPoints = 5;     // Paso del Trailing Stop en puntos
input bool           TrailSL = true;              // Activar Trailing Stop para Stop Loss
input bool           TrailTP = true;              // Activar Trailing Stop para Take Profit

// Parámetros específicos según el modo seleccionado
input long           TargetMagicNumber = 12345;   // Magic Number objetivo (si se selecciona ese modo)
input double         MinStopLossPoints = 35;     // Mínimo SL en puntos para filtrar (si se selecciona ese modo)
input double         MaxStopLossPoints = 65;     // Máximo SL en puntos para filtrar (si se selecciona ese modo)
input double         MinTakeProfitPoints = 100;   // Mínimo TP en puntos para filtrar (si se selecciona ese modo)
input double         MaxTakeProfitPoints = 150;   // Máximo TP en puntos para filtrar (si se selecciona ese modo)

// Parámetros de identificación del EA
input int            EAMagicNumber = 22222;       // Número mágico para identificar órdenes de este EA
input bool           ShowDebugInfo = false;       // Mostrar información de depuración

// Creamos un objeto para operaciones comerciales
CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Configuración inicial del objeto de trading
   trade.SetExpertMagicNumber(EAMagicNumber);
   
   // Verificar que los parámetros sean correctos
   if(TrailingStopPoints <= 0)
   {
      Print("Error: TrailingStopPoints debe ser mayor que cero");
      return INIT_PARAMETERS_INCORRECT;
   }
   
   if(TrailingStepPoints <= 0 || TrailingStepPoints > TrailingStopPoints)
   {
      Print("Error: TrailingStepPoints debe ser mayor que cero y menor o igual que TrailingStopPoints");
      return INIT_PARAMETERS_INCORRECT;
   }
   
   // Verificar parámetros específicos según el modo
   if(TrailingMode == BY_SL_TP_LEVELS)
   {
      if(MinStopLossPoints >= MaxStopLossPoints || MinTakeProfitPoints >= MaxTakeProfitPoints)
      {
         Print("Error: Los rangos de SL/TP no son válidos");
         return INIT_PARAMETERS_INCORRECT;
      }
   }
   
   // Mostrar información inicial
   Print("EA de Trailing Stop iniciado con éxito");
   Print("Símbolo actual: ", _Symbol);
   Print("Modo de trailing: ", GetTrailingModeName(TrailingMode));
   Print("Trailing Stop configurado a ", TrailingStopPoints, " puntos");
   Print("Trailing Step configurado a ", TrailingStepPoints, " puntos");
   
   if(TrailingMode == BY_MAGIC_NUMBER)
      Print("Filtrando órdenes con Magic Number: ", TargetMagicNumber);
   else if(TrailingMode == BY_SL_TP_LEVELS)
      Print("Filtrando órdenes con SL entre ", MinStopLossPoints, " y ", MaxStopLossPoints, 
            " puntos, y TP entre ", MinTakeProfitPoints, " y ", MaxTakeProfitPoints, " puntos");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Convierte el enum de modo a texto legible                        |
//+------------------------------------------------------------------+
string GetTrailingModeName(TRAILING_MODE mode)
{
   switch(mode)
   {
      case ALL_ORDERS: return "Todas las órdenes";
      case BY_MAGIC_NUMBER: return "Filtrado por Magic Number";
      case BY_SL_TP_LEVELS: return "Filtrado por niveles de SL/TP";
      default: return "Desconocido";
   }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("EA de Trailing Stop detenido. Razón: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Obtener precios actuales
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   // Aplicar trailing stop a las órdenes según el modo
   ManageOrders(ask, bid, point);
}

//+------------------------------------------------------------------+
//| Verifica si la orden cumple con los criterios de filtro          |
//+------------------------------------------------------------------+
bool ShouldManagePosition(ulong ticket, double point)
{
   // Si el modo es para todas las órdenes, siempre retornar verdadero
   if(TrailingMode == ALL_ORDERS)
      return true;
      
   // Si es filtrado por Magic Number
   if(TrailingMode == BY_MAGIC_NUMBER)
   {
      long position_magic = PositionGetInteger(POSITION_MAGIC);
      return (position_magic == TargetMagicNumber);
   }
   
   // Si es filtrado por niveles de SL/TP
   if(TrailingMode == BY_SL_TP_LEVELS)
   {
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      long position_type = PositionGetInteger(POSITION_TYPE);
      
      // Calculamos los puntos de SL y TP desde el precio de apertura
      double sl_points = 0;
      double tp_points = 0;
      
      // Para posiciones largas (compras)
      if(position_type == POSITION_TYPE_BUY)
      {
         if(sl > 0) sl_points = (open_price - sl) / point;
         if(tp > 0) tp_points = (tp - open_price) / point;
      }
      // Para posiciones cortas (ventas)
      else if(position_type == POSITION_TYPE_SELL)
      {
         if(sl > 0) sl_points = (sl - open_price) / point;
         if(tp > 0) tp_points = (open_price - tp) / point;
      }
      
      // Verificamos si los valores de SL y TP están dentro de los rangos especificados
      bool sl_in_range = (sl == 0) || (sl_points >= MinStopLossPoints && sl_points <= MaxStopLossPoints);
      bool tp_in_range = (tp == 0) || (tp_points >= MinTakeProfitPoints && tp_points <= MaxTakeProfitPoints);
      
      // Si el SL o TP está dentro del rango, consideramos la orden
      return (sl_in_range || tp_in_range);
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Función para gestionar todas las órdenes abiertas                |
//+------------------------------------------------------------------+
void ManageOrders(double ask, double bid, double point)
{
   // Recorremos todas las posiciones abiertas
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      // Seleccionamos la posición por índice
      ulong ticket = PositionGetTicket(i);
      
      // Si no se puede seleccionar la posición, continuamos con la siguiente
      if(!PositionSelectByTicket(ticket))
         continue;
         
      // Verificamos si la posición pertenece al símbolo actual
      string position_symbol = PositionGetString(POSITION_SYMBOL);
      if(position_symbol != _Symbol)
         continue;
      
      // Verificamos si la posición cumple con los criterios de filtrado
      if(!ShouldManagePosition(ticket, point))
         continue;
      
      // Obtenemos los datos de la posición
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
      long position_type = PositionGetInteger(POSITION_TYPE);
      
      // Mostrar información de depuración si está activado
      if(ShowDebugInfo)
      {
         long magic = PositionGetInteger(POSITION_MAGIC);
         Print("Analizando posición #", ticket, ", Magic: ", magic, 
               ", SL: ", sl, ", TP: ", tp, ", Tipo: ", position_type);
      }
      
      // Calcular nuevos niveles de SL y TP basados en trailing
      double new_sl = sl;
      double new_tp = tp;
      
      // Aplicar trailing stop para posiciones largas (compras)
      if(position_type == POSITION_TYPE_BUY)
      {
         // Trailing Stop Loss para posiciones largas
         if(TrailSL && current_price > open_price + TrailingStopPoints * point)
         {
            double trail_level = current_price - TrailingStopPoints * point;
            
            // Solo mover el SL si la nueva posición es mejor (más alta)
            if(trail_level > sl + TrailingStepPoints * point || sl == 0)
            {
               new_sl = NormalizeDouble(trail_level, _Digits);
            }
         }
         
         // Trailing Take Profit para posiciones largas
         if(TrailTP && tp != 0 && current_price > open_price)
         {
            double trail_tp = current_price + TrailingStopPoints * point;
            
            // Solo mover el TP si la nueva posición es mejor (más alta)
            if(trail_tp > tp + TrailingStepPoints * point)
            {
               new_tp = NormalizeDouble(trail_tp, _Digits);
            }
         }
      }
      // Aplicar trailing stop para posiciones cortas (ventas)
      else if(position_type == POSITION_TYPE_SELL)
      {
         // Trailing Stop Loss para posiciones cortas
         if(TrailSL && current_price < open_price - TrailingStopPoints * point)
         {
            double trail_level = current_price + TrailingStopPoints * point;
            
            // Solo mover el SL si la nueva posición es mejor (más baja)
            if(trail_level < sl - TrailingStepPoints * point || sl == 0)
            {
               new_sl = NormalizeDouble(trail_level, _Digits);
            }
         }
         
         // Trailing Take Profit para posiciones cortas
         if(TrailTP && tp != 0 && current_price < open_price)
         {
            double trail_tp = current_price - TrailingStopPoints * point;
            
            // Solo mover el TP si la nueva posición es mejor (más baja)
            if(trail_tp < tp - TrailingStepPoints * point)
            {
               new_tp = NormalizeDouble(trail_tp, _Digits);
            }
         }
      }
      
      // Si hay cambios, modificamos la posición
      if(new_sl != sl || new_tp != tp)
      {
         trade.PositionModify(ticket, new_sl, new_tp);
         
         if(GetLastError() == 0)
            Print("Posición #", ticket, " modificada: SL=", new_sl, ", TP=", new_tp);
         else
            Print("Error al modificar posición #", ticket, ": ", GetLastError());
      }
   }
}

//+------------------------------------------------------------------+
//| Función para gestionar eventos de nueva orden                     |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                       const MqlTradeRequest& request,
                       const MqlTradeResult& result)
{
   // Monitoreo de nuevas transacciones de trading para posible acción inmediata
   // Verificar el tipo de transacción
   // TRADE_TRANSACTION_ORDER_ADD (0) - Se añadió una nueva orden
   // TRADE_TRANSACTION_POSITION (6) - Cambio en una posición existente
   // TRADE_TRANSACTION_REQUEST (10) - Notificación de que se ha procesado una solicitud comercial
   if(trans.type == TRADE_TRANSACTION_ORDER_ADD || trans.type == TRADE_TRANSACTION_POSITION)
   {
      // Podemos procesar inmediatamente las nuevas órdenes si es necesario
      if(ShowDebugInfo)
         Print("Nueva transacción detectada: ", EnumToString(trans.type), ", Ticket: ", trans.order);
   }
}
//+------------------------------------------------------------------+