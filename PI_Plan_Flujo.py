"""
═══════════════════════════════════════════════════════════════════════════════
PI_Plan_Flujo.py - VERSIÓN CON CONSTRUCCIÓN DE RUTAS + MÓDULO SIG
═══════════════════════════════════════════════════════════════════════════════

Incluye:
1. NUEVO: Submódulo SIG para georeferenciación de redes
2. Modelo AIMMS con 8 restricciones
3. Algoritmo de construcción de rutas con familias indivisibles
4. Casos de prueba simples y complejos

NOTA: PI_Plan_Flujo_SIMPLE_FUNCIONAL.py contiene la versión base validada
═══════════════════════════════════════════════════════════════════════════════
"""

from ortools.linear_solver import pywraplp
from collections import defaultdict, deque
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# SUBMÓDULO SIG: SISTEMA DE INFORMACIÓN GEOGRÁFICA PARA REDES DE EVACUACIÓN
# ═══════════════════════════════════════════════════════════════════════════
"""
Este submódulo permite dos modos de operación para la matriz de distancias d_i,j:

MODO 1 - MANUAL (por defecto):
    Los arcos se definen directamente con distancias conocidas.
    Ejemplo: arcos = {('A1', 'R1'): 5.0, ('R1', 'F1'): 3.0}

MODO 2 - GEOREFERENCIADO:
    Los nodos se asocian a ubicaciones geográficas reales.
    Las distancias se calculan automáticamente usando:
    - Haversine (línea recta geodésica) - siempre disponible
    - OSRM/OpenRouteService (ruta real por carretera) - requiere conexión
    
Uso típico:
    # Crear preprocesador
    geo = PreprocesadorGeoRed(modo='georeferenciado')
    
    # Definir ubicaciones
    geo.agregar_nodo('A1', 'Vedado, La Habana, Cuba', tipo='A')
    geo.agregar_nodo('R1', 'Cotorro, La Habana, Cuba', tipo='R')
    geo.agregar_nodo('F1', 'Santa María del Mar, Cuba', tipo='F')
    
    # Generar matriz de distancias automáticamente
    arcos = geo.generar_arcos()
    
    # Ver tabla de mapeo
    geo.imprimir_tabla_red()
"""


class ModoDistancia(Enum):
    """Modos de cálculo de distancia"""
    MANUAL = 'manual'
    HAVERSINE = 'haversine'
    OSRM = 'osrm'
    ORS = 'openrouteservice'


@dataclass
class NodoGeo:
    """Representa un nodo con información geográfica"""
    id: str
    nombre: str
    tipo: str  # 'A' = salida, 'R' = tránsito, 'F' = refugio
    lat: Optional[float] = None
    lon: Optional[float] = None
    direccion: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def tiene_coordenadas(self) -> bool:
        return self.lat is not None and self.lon is not None
    
    def coordenadas(self) -> Tuple[float, float]:
        if not self.tiene_coordenadas():
            raise ValueError(f"Nodo {self.id} no tiene coordenadas definidas")
        return (self.lat, self.lon)


class GeoLocalizador:
    """
    Servicio de geocodificación y cálculo de distancias.
    
    Soporta:
    - Geocodificación directa (coordenadas manuales)
    - Geocodificación por nombre (requiere geopy)
    - Distancia Haversine (línea recta geodésica)
    - Distancia por ruta real (requiere API externa)
    """
    
    # Radio de la Tierra en kilómetros
    RADIO_TIERRA_KM = 6371.0
    
    def __init__(self, usar_cache: bool = True):
        self.cache_coordenadas: Dict[str, Tuple[float, float]] = {}
        self.cache_distancias: Dict[Tuple[str, str], float] = {}
        self.usar_cache = usar_cache
        self._geocoder = None
        
    def _inicializar_geocoder(self):
        """Inicializa el geocoder de geopy si está disponible"""
        if self._geocoder is None:
            try:
                from geopy.geocoders import Nominatim
                self._geocoder = Nominatim(user_agent="evacuacion_model_v1")
            except ImportError:
                print("⚠️  geopy no instalado. Use: pip install geopy")
                print("    Geocodificación por nombre no disponible.")
                self._geocoder = False
        return self._geocoder
    
    def geocodificar(self, direccion: str) -> Optional[Tuple[float, float]]:
        """
        Obtiene coordenadas (lat, lon) de una dirección.
        
        Args:
            direccion: Dirección textual (ej: "Vedado, La Habana, Cuba")
            
        Returns:
            Tupla (latitud, longitud) o None si no se encuentra
        """
        # Verificar cache
        if self.usar_cache and direccion in self.cache_coordenadas:
            return self.cache_coordenadas[direccion]
        
        geocoder = self._inicializar_geocoder()
        if not geocoder:
            return None
            
        try:
            ubicacion = geocoder.geocode(direccion, timeout=10)
            if ubicacion:
                coords = (ubicacion.latitude, ubicacion.longitude)
                if self.usar_cache:
                    self.cache_coordenadas[direccion] = coords
                return coords
        except Exception as e:
            print(f"⚠️  Error geocodificando '{direccion}': {e}")
        
        return None
    
    @staticmethod
    def distancia_haversine(
        lat1: float, lon1: float, 
        lat2: float, lon2: float
    ) -> float:
        """
        Calcula distancia geodésica usando fórmula de Haversine.
        
        La fórmula de Haversine calcula la distancia del arco de círculo
        máximo entre dos puntos en una esfera (la Tierra).
        
        d = 2r × arcsin(√(sin²(Δφ/2) + cos(φ₁)×cos(φ₂)×sin²(Δλ/2)))
        
        donde:
            φ = latitud en radianes
            λ = longitud en radianes
            r = radio de la Tierra (6371 km)
        
        Args:
            lat1, lon1: Coordenadas del punto origen (grados)
            lat2, lon2: Coordenadas del punto destino (grados)
            
        Returns:
            Distancia en kilómetros
        """
        # Convertir a radianes
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Fórmula de Haversine
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return GeoLocalizador.RADIO_TIERRA_KM * c
    
    def distancia_ruta_osrm(
        self, 
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        servidor: str = "http://router.project-osrm.org"
    ) -> Optional[float]:
        """
        Calcula distancia real por carretera usando OSRM.
        
        OSRM (Open Source Routing Machine) proporciona rutas reales
        por carretera, considerando la red vial existente.
        
        Args:
            lat1, lon1: Coordenadas origen
            lat2, lon2: Coordenadas destino
            servidor: URL del servidor OSRM
            
        Returns:
            Distancia en kilómetros o None si falla
        """
        try:
            import requests
        except ImportError:
            print("⚠️  requests no instalado. Use: pip install requests")
            return None
            
        url = f"{servidor}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
        params = {"overview": "false"}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "Ok":
                    # Distancia en metros -> kilómetros
                    return data["routes"][0]["distance"] / 1000.0
        except Exception as e:
            print(f"⚠️  Error OSRM: {e}")
        
        return None
    
    def calcular_distancia(
        self,
        origen: NodoGeo,
        destino: NodoGeo,
        modo: ModoDistancia = ModoDistancia.HAVERSINE,
        factor_ajuste: float = 1.3
    ) -> float:
        """
        Calcula distancia entre dos nodos.
        
        Args:
            origen: Nodo de origen
            destino: Nodo de destino
            modo: Método de cálculo (HAVERSINE, OSRM, etc.)
            factor_ajuste: Factor para aproximar distancia por carretera
                          desde distancia Haversine (típico: 1.2-1.4)
        
        Returns:
            Distancia en kilómetros
        """
        if not origen.tiene_coordenadas() or not destino.tiene_coordenadas():
            raise ValueError("Ambos nodos deben tener coordenadas")
        
        # Verificar cache
        cache_key = (origen.id, destino.id, modo.value)
        if self.usar_cache and cache_key in self.cache_distancias:
            return self.cache_distancias[cache_key]
        
        lat1, lon1 = origen.coordenadas()
        lat2, lon2 = destino.coordenadas()
        
        if modo == ModoDistancia.HAVERSINE:
            dist = self.distancia_haversine(lat1, lon1, lat2, lon2)
            # Aplicar factor de ajuste para aproximar distancia real
            dist *= factor_ajuste
            
        elif modo == ModoDistancia.OSRM:
            dist = self.distancia_ruta_osrm(lat1, lon1, lat2, lon2)
            if dist is None:
                # Fallback a Haversine si OSRM falla
                print(f"    Fallback Haversine para {origen.id}->{destino.id}")
                dist = self.distancia_haversine(lat1, lon1, lat2, lon2) * factor_ajuste
        else:
            dist = self.distancia_haversine(lat1, lon1, lat2, lon2) * factor_ajuste
        
        # Guardar en cache
        if self.usar_cache:
            self.cache_distancias[cache_key] = dist
        
        return dist


class PreprocesadorGeoRed:
    """
    Preprocesador de redes de evacuación con soporte SIG.
    
    Permite construir la red de evacuación de dos formas:
    
    1. MODO MANUAL (tradicional):
       - Los arcos se definen directamente con sus distancias
       - No requiere coordenadas geográficas
       - Compatible con el flujo existente
       
    2. MODO GEOREFERENCIADO:
       - Los nodos se definen con coordenadas (lat, lon) o direcciones
       - Las distancias se calculan automáticamente
       - Genera tabla de mapeo nodo -> ubicación real
    """
    
    def __init__(
        self, 
        modo: str = 'manual',
        metodo_distancia: ModoDistancia = ModoDistancia.HAVERSINE,
        factor_ajuste: float = 1.3
    ):
        """
        Inicializa el preprocesador.
        
        Args:
            modo: 'manual' o 'georeferenciado'
            metodo_distancia: Método para calcular distancias
            factor_ajuste: Factor para ajustar Haversine a distancia real
        """
        self.modo = modo
        self.metodo_distancia = metodo_distancia
        self.factor_ajuste = factor_ajuste
        
        self.nodos: Dict[str, NodoGeo] = {}
        self.arcos_definidos: Dict[Tuple[str, str], float] = {}
        self.conectividad: Dict[str, List[str]] = defaultdict(list)
        
        self.geolocalizador = GeoLocalizador() if modo == 'georeferenciado' else None
    
    def agregar_nodo(
        self,
        id_nodo: str,
        nombre_o_direccion: str,
        tipo: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> NodoGeo:
        """
        Agrega un nodo a la red.
        
        Args:
            id_nodo: Identificador único (ej: 'A1', 'R2', 'F1')
            nombre_o_direccion: Nombre descriptivo o dirección para geocodificar
            tipo: 'A' (salida), 'R' (tránsito), 'F' (refugio)
            lat, lon: Coordenadas (opcional si se proporciona dirección)
            metadata: Información adicional
            
        Returns:
            NodoGeo creado
        """
        if tipo not in ('A', 'R', 'F'):
            raise ValueError(f"Tipo debe ser 'A', 'R' o 'F', recibido: {tipo}")
        
        nodo = NodoGeo(
            id=id_nodo,
            nombre=nombre_o_direccion,
            tipo=tipo,
            lat=lat,
            lon=lon,
            direccion=nombre_o_direccion if lat is None else None,
            metadata=metadata or {}
        )
        
        # Si no hay coordenadas y estamos en modo georeferenciado, geocodificar
        if self.modo == 'georeferenciado' and not nodo.tiene_coordenadas():
            coords = self.geolocalizador.geocodificar(nombre_o_direccion)
            if coords:
                nodo.lat, nodo.lon = coords
                print(f"✓ Geocodificado {id_nodo}: {coords}")
            else:
                print(f"⚠️  No se pudo geocodificar {id_nodo}: {nombre_o_direccion}")
        
        self.nodos[id_nodo] = nodo
        return nodo
    
    def agregar_nodo_coordenadas(
        self,
        id_nodo: str,
        nombre: str,
        tipo: str,
        lat: float,
        lon: float
    ) -> NodoGeo:
        """
        Agrega un nodo con coordenadas directas (sin geocodificación).
        
        Args:
            id_nodo: Identificador único
            nombre: Nombre descriptivo del lugar
            tipo: 'A', 'R' o 'F'
            lat, lon: Coordenadas geográficas
            
        Returns:
            NodoGeo creado
        """
        return self.agregar_nodo(id_nodo, nombre, tipo, lat=lat, lon=lon)
    
    def definir_arco_manual(
        self,
        origen: str,
        destino: str,
        distancia: float
    ):
        """
        Define un arco con distancia manual (modo tradicional).
        
        Args:
            origen: ID nodo origen
            destino: ID nodo destino
            distancia: Distancia en km
        """
        self.arcos_definidos[(origen, destino)] = distancia
        self.conectividad[origen].append(destino)
    
    def definir_conectividad(
        self,
        origen: str,
        destinos: List[str]
    ):
        """
        Define qué nodos están conectados desde un origen.
        Solo es necesario en modo georeferenciado para especificar
        qué arcos deben existir (las distancias se calculan automáticamente).
        
        Args:
            origen: ID del nodo origen
            destinos: Lista de IDs de nodos destino
        """
        self.conectividad[origen].extend(destinos)
    
    def generar_arcos(
        self,
        conectividad_completa: bool = False
    ) -> Dict[Tuple[str, str], float]:
        """
        Genera el diccionario de arcos con distancias.
        
        Args:
            conectividad_completa: Si True, conecta todos los nodos
                                  compatibles (A->R, A->F, R->R, R->F)
        
        Returns:
            Dict[(origen, destino)] = distancia_km
        """
        arcos = {}
        
        if self.modo == 'manual':
            # En modo manual, retornar arcos definidos directamente
            return dict(self.arcos_definidos)
        
        # Modo georeferenciado
        if conectividad_completa:
            self._generar_conectividad_completa()
        
        # Calcular distancias para cada arco definido en conectividad
        for origen_id, destinos in self.conectividad.items():
            if origen_id not in self.nodos:
                print(f"⚠️  Nodo origen {origen_id} no definido")
                continue
            origen = self.nodos[origen_id]
            
            for destino_id in destinos:
                if destino_id not in self.nodos:
                    print(f"⚠️  Nodo destino {destino_id} no definido")
                    continue
                destino = self.nodos[destino_id]
                
                # Verificar que ambos tienen coordenadas
                if not origen.tiene_coordenadas() or not destino.tiene_coordenadas():
                    print(f"⚠️  Sin coordenadas para arco {origen_id}->{destino_id}")
                    continue
                
                # Calcular distancia
                dist = self.geolocalizador.calcular_distancia(
                    origen, destino,
                    modo=self.metodo_distancia,
                    factor_ajuste=self.factor_ajuste
                )
                arcos[(origen_id, destino_id)] = round(dist, 2)
        
        return arcos
    
    def _generar_conectividad_completa(self):
        """Genera conectividad automática según tipos de nodos"""
        nodos_a = [n.id for n in self.nodos.values() if n.tipo == 'A']
        nodos_r = [n.id for n in self.nodos.values() if n.tipo == 'R']
        nodos_f = [n.id for n in self.nodos.values() if n.tipo == 'F']
        
        # A -> R y A -> F
        for a in nodos_a:
            self.conectividad[a].extend(nodos_r)
            self.conectividad[a].extend(nodos_f)
        
        # R -> R y R -> F
        for r in nodos_r:
            self.conectividad[r].extend([x for x in nodos_r if x != r])
            self.conectividad[r].extend(nodos_f)
    
    def obtener_listas_nodos(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Retorna las listas de nodos separadas por tipo.
        
        Returns:
            (nodos_salida, nodos_transito, nodos_llegada)
        """
        nodos_salida = [n.id for n in self.nodos.values() if n.tipo == 'A']
        nodos_transito = [n.id for n in self.nodos.values() if n.tipo == 'R']
        nodos_llegada = [n.id for n in self.nodos.values() if n.tipo == 'F']
        return nodos_salida, nodos_transito, nodos_llegada
    
    def imprimir_tabla_red(self):
        """Imprime tabla de mapeo nodos -> ubicaciones"""
        print("\n" + "═" * 80)
        print("📍 TABLA DE RED GEOREFERENCIADA")
        print("═" * 80)
        
        # Encabezado
        print(f"{'ID':<8} {'Tipo':<6} {'Nombre/Dirección':<30} {'Lat':>12} {'Lon':>12}")
        print("─" * 80)
        
        # Ordenar por tipo y luego por ID
        orden_tipo = {'A': 0, 'R': 1, 'F': 2}
        nodos_ordenados = sorted(
            self.nodos.values(), 
            key=lambda n: (orden_tipo[n.tipo], n.id)
        )
        
        for nodo in nodos_ordenados:
            tipo_str = {'A': 'Salida', 'R': 'Tráns.', 'F': 'Refug.'}[nodo.tipo]
            lat_str = f"{nodo.lat:.6f}" if nodo.lat else "N/D"
            lon_str = f"{nodo.lon:.6f}" if nodo.lon else "N/D"
            nombre = nodo.nombre[:28] + ".." if len(nodo.nombre) > 30 else nodo.nombre
            print(f"{nodo.id:<8} {tipo_str:<6} {nombre:<30} {lat_str:>12} {lon_str:>12}")
        
        print("─" * 80)
        print(f"Total: {len(self.nodos)} nodos")
    
    def imprimir_matriz_distancias(self, arcos: Dict[Tuple[str, str], float]):
        """Imprime matriz de distancias calculadas"""
        print("\n" + "═" * 60)
        print("📏 MATRIZ DE DISTANCIAS d(i,j) [km]")
        print("═" * 60)
        
        for (i, j), dist in sorted(arcos.items()):
            nombre_i = self.nodos.get(i, NodoGeo(i, i, 'A')).nombre[:15]
            nombre_j = self.nodos.get(j, NodoGeo(j, j, 'F')).nombre[:15]
            print(f"  {i} → {j}  :  {dist:>8.2f} km  ({nombre_i} → {nombre_j})")
        
        print("─" * 60)
        print(f"Total: {len(arcos)} arcos")
    
    def exportar_geojson(self, arcos: Dict[Tuple[str, str], float]) -> Dict:
        """
        Exporta la red en formato GeoJSON para visualización.
        
        Returns:
            Diccionario GeoJSON con puntos (nodos) y líneas (arcos)
        """
        features = []
        
        # Agregar nodos como puntos
        for nodo in self.nodos.values():
            if nodo.tiene_coordenadas():
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [nodo.lon, nodo.lat]
                    },
                    "properties": {
                        "id": nodo.id,
                        "nombre": nodo.nombre,
                        "tipo": nodo.tipo,
                        "tipo_desc": {'A': 'Zona Salida', 'R': 'Punto Tránsito', 'F': 'Refugio'}[nodo.tipo]
                    }
                }
                features.append(feature)
        
        # Agregar arcos como líneas
        for (i, j), dist in arcos.items():
            if i in self.nodos and j in self.nodos:
                ni, nj = self.nodos[i], self.nodos[j]
                if ni.tiene_coordenadas() and nj.tiene_coordenadas():
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [ni.lon, ni.lat],
                                [nj.lon, nj.lat]
                            ]
                        },
                        "properties": {
                            "origen": i,
                            "destino": j,
                            "distancia_km": dist
                        }
                    }
                    features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE TRANSFERENCIA: Conversión Red Abstracta <-> Red Georeferenciada
# ═══════════════════════════════════════════════════════════════════════════

def crear_red_manual(
    nodos_salida: List[str],
    nodos_transito: List[str],
    nodos_llegada: List[str],
    arcos: Dict[Tuple[str, str], float]
) -> PreprocesadorGeoRed:
    """
    Crea un PreprocesadorGeoRed desde una definición manual tradicional.
    Permite usar el mismo objeto para ambos modos.
    
    Args:
        nodos_salida: Lista de IDs de nodos tipo A
        nodos_transito: Lista de IDs de nodos tipo R
        nodos_llegada: Lista de IDs de nodos tipo F
        arcos: Diccionario {(origen, destino): distancia}
        
    Returns:
        PreprocesadorGeoRed configurado
    """
    geo = PreprocesadorGeoRed(modo='manual')
    
    for ns in nodos_salida:
        geo.agregar_nodo(ns, ns, 'A')
    for nt in nodos_transito:
        geo.agregar_nodo(nt, nt, 'R')
    for nf in nodos_llegada:
        geo.agregar_nodo(nf, nf, 'F')
    
    for (i, j), dist in arcos.items():
        geo.definir_arco_manual(i, j, dist)
    
    return geo


def crear_red_georeferenciada_ejemplo_habana() -> Tuple[PreprocesadorGeoRed, Dict]:
    """
    Red georeferenciada para La Habana - COMPATIBLE CON PII_Flota_Asig.py
    Estructura: A1, A2 -> R1 -> F1, F2
    
    Returns:
        (PreprocesadorGeoRed, arcos_calculados)
    """
    print("\n" + "▓" * 70)
    print("▓" + " " * 15 + "EJEMPLO SIG: LA HABANA (Compatible PII)" + " " * 13 + "▓")
    print("▓" * 70)
    
    geo = PreprocesadorGeoRed(
        modo='georeferenciado',
        metodo_distancia=ModoDistancia.HAVERSINE,
        factor_ajuste=1.3
    )
    
    # ZONAS DE SALIDA (A) - Solo A1, A2 para compatibilidad con PII
    geo.agregar_nodo_coordenadas('A1', 'Vedado (La Habana)', 'A', 
                                  lat=23.1367, lon=-82.4047)
    geo.agregar_nodo_coordenadas('A2', 'Miramar (Playa)', 'A', 
                                  lat=23.1165, lon=-82.4318)
    
    # PUNTO DE TRÁNSITO (R) - Solo R1
    geo.agregar_nodo_coordenadas('R1', 'Plaza de la Revolución', 'R', 
                                  lat=23.1174, lon=-82.4118)
    
    # REFUGIOS (F)
    geo.agregar_nodo_coordenadas('F1', 'Cotorro', 'F', 
                                  lat=23.0280, lon=-82.2791)
    geo.agregar_nodo_coordenadas('F2', 'San José de las Lajas', 'F', 
                                  lat=22.9614, lon=-82.1514)
    
    # Conectividad: A -> R1 -> F (estructura del caso simple PII)
    geo.definir_conectividad('A1', ['R1'])
    geo.definir_conectividad('A2', ['R1'])
    geo.definir_conectividad('R1', ['F1', 'F2'])
    
    arcos = geo.generar_arcos()
    
    geo.imprimir_tabla_red()
    geo.imprimir_matriz_distancias(arcos)
    
    return geo, arcos


# ═══════════════════════════════════════════════════════════════════════════
# GENERADOR DE IDF (sin cambios)
# ═══════════════════════════════════════════════════════════════════════════

def generar_idf(fi_nominal):
    """Genera IDF: 1 ID por cada (h, ns) con familias > 0"""
    idf = {}
    id_counter = 0
    for (h, ns), cantidad in fi_nominal.items():
        if cantidad > 0:
            id_counter += 1
            idf[(id_counter, h, ns)] = cantidad
    return idf


# ═══════════════════════════════════════════════════════════════════════════
# MODELO AIMMS (sin cambios)
# ═══════════════════════════════════════════════════════════════════════════

class ModeloAIMMS:
    """Modelo de evacuación AIMMS"""
    
    def __init__(self, nodos_salida, nodos_transito, nodos_llegada, arcos, idf, capacidades):
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self.solver:
            raise Exception('No se pudo crear solver SCIP')
        
        self.nodos_salida = nodos_salida
        self.nodos_transito = nodos_transito
        self.nodos_llegada = nodos_llegada
        self.arcos = arcos
        self.idf = idf
        self.capacidades = capacidades
        
        # Calcular P (total personas) automáticamente desde idf
        self.P = sum(h * q for (_, h, _), q in idf.items())
        
        self.X = {}
        self.Y = {}
        
    def crear_variables(self):
        """Crea variables X e Y"""
        for (id_val, h, origen), cantidad in self.idf.items():
            for (i, j) in self.arcos.keys():
                self.X[(id_val, h, i, j)] = self.solver.IntVar(
                    0, cantidad, f'X_{id_val}_{h}_{i}_{j}'
                )
        
        for nodo in self.nodos_transito + self.nodos_llegada:
            self.Y[nodo] = self.solver.BoolVar(f'Y_{nodo}')
    
    def agregar_restricciones(self):
        """8 restricciones AIMMS"""
        rf_nodes = self.nodos_transito + self.nodos_llegada
        ar_nodes = self.nodos_salida + self.nodos_transito
        
        # 1. Robust_Salidas_C1 - TODOS los nodos A con TODOS los IDs
        for ns in self.nodos_salida:
            for (id_val, h, origen), cantidad_idf in self.idf.items():
                rhs = cantidad_idf if ns == origen else 0
                salidas = [self.X[(id_val, h, ns, rf)] 
                          for rf in rf_nodes if (id_val, h, ns, rf) in self.X]
                entradas = [self.X[(id_val, h, rf, ns)] 
                           for rf in rf_nodes if (id_val, h, rf, ns) in self.X]
                if salidas or entradas:
                    self.solver.Add(sum(salidas) - sum(entradas) == rhs)
        
        # 2. capac_llegada_origen (usa α específico por nodo o P)
        for ns in self.nodos_salida:
            alpha = self.capacidades.get(('alpha', ns), self.P)
            personas = [h * self.X[(id_val, h, i, j)] 
                       for (id_val, h, i, j) in self.X.keys() if i == ns]
            if personas:
                self.solver.Add(sum(personas) <= alpha)
        
        # 3. Robust_Flujo_Transito
        for nt in self.nodos_transito:
            for (id_val, h, _) in self.idf.keys():
                todos_nodos = self.nodos_salida + self.nodos_transito + self.nodos_llegada
                entradas = [self.X[(id_val, h, i, nt)] 
                           for i in todos_nodos if (id_val, h, i, nt) in self.X]
                salidas = [self.X[(id_val, h, nt, j)] 
                          for j in todos_nodos if (id_val, h, nt, j) in self.X]
                if entradas or salidas:
                    self.solver.Add(sum(entradas) - sum(salidas) == 0)
        
        # 4. cap_llegada_punto_transito (usa β específico por nodo o P)
        for nt in self.nodos_transito:
            beta = self.capacidades.get(('beta', nt), self.P)
            personas = [h * self.X[(id_val, h, j, nt_val)] 
                       for (id_val, h, j, nt_val) in self.X.keys() if nt_val == nt]
            if personas:
                self.solver.Add(sum(personas) <= beta)
        
        # 5. Robust_Llegada_C2 (capacidad neta refugios π)
        for nll in self.nodos_llegada:
            pi = self.capacidades.get(('pi', nll), float('inf'))
            if pi < float('inf'):
                entradas = [h * self.X[(id_val, h, i, j)] 
                           for (id_val, h, i, j) in self.X.keys() if j == nll]
                salidas = [h * self.X[(id_val, h, i, j)] 
                          for (id_val, h, i, j) in self.X.keys() if i == nll]
                if entradas or salidas:
                    self.solver.Add(sum(entradas) - sum(salidas) <= pi)
        
        # 5b. ACTIVACIÓN DE Y[j] - Si hay flujo hacia refugio F, Y[F] = 1
        # Restricción: Σᵢₕ X[id,h,i,j] ≤ M × Y[j] (para refugios F)
        # Esto asegura que el costo de acondicionamiento c_j se contabilice
        M = self.P  # BigM = total de personas a evacuar
        for nll in self.nodos_llegada:
            flujos_entrada = [self.X[(id_val, h, i, j)] 
                             for (id_val, h, i, j) in self.X.keys() if j == nll]
            if flujos_entrada and nll in self.Y:
                # Si hay cualquier flujo hacia el refugio, Y[nll] debe ser 1
                self.solver.Add(sum(flujos_entrada) <= M * self.Y[nll])
        
        # 5c. ACTIVACIÓN DE Y[j] - Para nodos de tránsito R
        for nt in self.nodos_transito:
            flujos_entrada = [self.X[(id_val, h, i, j)] 
                             for (id_val, h, i, j) in self.X.keys() if j == nt]
            if flujos_entrada and nt in self.Y:
                self.solver.Add(sum(flujos_entrada) <= M * self.Y[nt])
        
        # 6. cap_llegada_centro_seguro (usa γ específico por nodo o P)
        for nll in self.nodos_llegada:
            gamma = self.capacidades.get(('gamma', nll), self.P)
            personas = [h * self.X[(id_val, h, i, j)] 
                       for (id_val, h, i, j) in self.X.keys() if j == nll]
            if personas:
                self.solver.Add(sum(personas) <= gamma)
        
        # 7. Equilibrio1
        for (id_val, h, origen) in self.idf.keys():
            izq_salidas = [self.X[(id_val, h, ns, rf)] 
                          for ns in self.nodos_salida for rf in rf_nodes 
                          if (id_val, h, ns, rf) in self.X]
            izq_entradas = [self.X[(id_val, h, rf, ns)] 
                           for ns in self.nodos_salida for rf in rf_nodes 
                           if (id_val, h, rf, ns) in self.X]
            der_salidas = [self.X[(id_val, h, ar, nll)] 
                          for ar in ar_nodes for nll in self.nodos_llegada 
                          if (id_val, h, ar, nll) in self.X]
            der_entradas = [self.X[(id_val, h, nll, ar)] 
                           for ar in ar_nodes for nll in self.nodos_llegada 
                           if (id_val, h, nll, ar) in self.X]
            
            if izq_salidas or izq_entradas or der_salidas or der_entradas:
                self.solver.Add(
                    sum(izq_salidas) - sum(izq_entradas) == 
                    sum(der_salidas) - sum(der_entradas)
                )
        
        # 8. Equilibrio2
        for nll in self.nodos_llegada:
            for (id_val, h, _) in self.idf.keys():
                todos_nodos = self.nodos_salida + self.nodos_transito + self.nodos_llegada
                salidas = [self.X[(id_val, h, nll, j)] 
                          for j in todos_nodos if (id_val, h, nll, j) in self.X]
                entradas = [self.X[(id_val, h, j, nll)] 
                           for j in todos_nodos if (id_val, h, j, nll) in self.X]
                if salidas and entradas:
                    self.solver.Add(sum(salidas) <= sum(entradas))
    
    def establecer_objetivo(self, costo_por_km=0.36, costos_nodos=None):
        """
        Función objetivo: min Z = Σᵢⱼ c·dᵢⱼ·h·xᵢⱼ + Σⱼ cⱼ·yⱼ
        
        Args:
            costo_por_km: Costo unitario de transporte ($/km·persona)
            costos_nodos: Dict {nodo: costo_fijo} para R y F. Default=1.0 si no se especifica.
        """
        objetivo = self.solver.Objective()
        objetivo.SetMinimization()
        
        # Término 1: Costo de transporte (c * d_ij * h * x_ij)
        for (id_val, h, i, j), var in self.X.items():
            distancia = self.arcos.get((i, j), 0)
            if distancia > 0:
                coef = costo_por_km * distancia * h
                objetivo.SetCoefficient(var, coef)
        
        # Término 2: Costo fijo de nodos (c_j * y_j)
        for nodo, var in self.Y.items():
            if costos_nodos and nodo in costos_nodos:
                costo_nodo = costos_nodos[nodo]
            else:
                costo_nodo = 1.0  # Valor por defecto
            objetivo.SetCoefficient(var, costo_nodo)
    
    def resolver(self, tiempo_limite=300):
        """Resuelve el modelo"""
        self.solver.SetTimeLimit(tiempo_limite * 1000)
        status = self.solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            return 'OPTIMAL'
        elif status == pywraplp.Solver.FEASIBLE:
            return 'FEASIBLE'
        else:
            return 'INFEASIBLE'
    
    def obtener_solucion(self):
        """Extrae la solución"""
        X_sol = {}
        for (id_val, h, i, j), var in self.X.items():
            valor = var.solution_value()
            if valor > 0.5:
                X_sol[(id_val, h, i, j)] = int(valor)
        
        Y_sol = {nodo: int(var.solution_value()) for nodo, var in self.Y.items()}
        
        return {
            'X': X_sol,
            'Y': Y_sol,
            'objetivo': self.solver.Objective().Value(),
            'tiempo': self.solver.WallTime() / 1000.0
        }


# ═══════════════════════════════════════════════════════════════════════════
# ALGORITMO DE CONSTRUCCIÓN DE RUTAS (sin cambios)
# ═══════════════════════════════════════════════════════════════════════════

class ConstructorRutas:
    """
    Construye rutas completas desde origen hasta destino final
    Maneja familias indivisibles y nodos intermedios
    
    ESTRATEGIA:
    - Para cada ID, explorar TODAS las rutas posibles desde su origen
    - Rastrear cuántas familias toman cada ruta
    - NO recontar: cada familia se asigna a UNA sola ruta completa
    """
    
    def __init__(self, X_sol, idf, nodos_salida, nodos_transito, nodos_llegada, arcos, 
                 distancias_arcos=None):
        self.X = X_sol
        self.idf = idf
        self.nodos_salida = nodos_salida
        self.nodos_transito = nodos_transito
        self.nodos_llegada = nodos_llegada
        self.arcos = arcos
        
        # Distancias de arcos (dict {(i,j): dist_km} o None)
        self.distancias_arcos = distancias_arcos if distancias_arcos else {}
        
        # Diccionario de rutas
        self.rutas = {}
        self.contador_rutas = 0
        
    def construir_rutas(self):
        """Construye todas las rutas para cada ID"""
        print("\n" + "="*80)
        print("🛤️  CONSTRUCCIÓN DE RUTAS")
        print("="*80)
        
        for (id_val, h, origen), cantidad_total in self.idf.items():
            print(f"\n📍 ID={id_val}, h={h}, origen={origen} ({cantidad_total} familias)")
            
            # Construir rutas desde el origen hasta refugios finales
            rutas_id = self._construir_rutas_id(id_val, h, origen, cantidad_total)
            
            # Validar
            familias_contadas = sum(r['familias'] for r in rutas_id)
            if abs(familias_contadas - cantidad_total) > 0.01:
                print(f"   ⚠️  ERROR: {cantidad_total} esperadas, {familias_contadas} contadas")
            else:
                print(f"   ✅ {len(rutas_id)} rutas, {familias_contadas} familias ✓")
        
        return self.rutas
    
    def _construir_rutas_id(self, id_val, h, origen, cantidad_total):
        """Construye rutas para un ID específico usando DFS"""
        rutas_encontradas = []
        
        # Crear grafo de flujos para este ID
        grafo = defaultdict(dict)
        for (id_x, h_x, i, j), familias in self.X.items():
            if id_x == id_val and h_x == h:
                grafo[i][j] = familias
        
        # Explorar rutas desde origen
        self._explorar_rutas(id_val, h, origen, [origen], grafo, rutas_encontradas)
        
        return rutas_encontradas
    
    def _explorar_rutas(self, id_val, h, origen, camino, grafo, rutas_encontradas):
        """
        DFS para explorar todas las rutas desde origen hasta refugios
        
        Args:
            id_val, h: identificadores
            origen: nodo origen original
            camino: lista de nodos en el camino actual
            grafo: dict[nodo_origen][nodo_destino] = familias
            rutas_encontradas: lista para guardar rutas
        """
        nodo_actual = camino[-1]
        
        # Si es un refugio, registrar ruta
        if nodo_actual in self.nodos_llegada:
            # Calcular cuántas familias llegan vs cuántas salen
            familias_entran = self._calcular_flujo_entrante(camino, grafo)
            familias_salen = sum(grafo[nodo_actual].values()) if nodo_actual in grafo else 0
            familias_quedan = familias_entran - familias_salen
            
            if familias_quedan > 0.5:  # Threshold para evitar errores numéricos
                self.contador_rutas += 1
                ruta_str = "->".join(camino)
                
                # Calcular distancia total de la ruta en km
                distancia_km = 0.0
                for idx in range(len(camino) - 1):
                    arco = (camino[idx], camino[idx + 1])
                    distancia_km += self.distancias_arcos.get(arco, 0.0)
                
                ruta_info = {
                    'id_ruta': self.contador_rutas,
                    'ruta': ruta_str,
                    'familias': int(familias_quedan),
                    'personas': int(familias_quedan) * h,
                    'tamano_familia': h,
                    'id_familia': id_val,
                    'origen': origen,
                    'destino': nodo_actual,
                    'longitud': len(camino) - 1,
                    'distancia_km': distancia_km,  # NUEVO: distancia real en km
                    'nodos_intermedios': camino[1:-1]
                }
                
                self.rutas[self.contador_rutas] = ruta_info
                rutas_encontradas.append(ruta_info)
                
                # Mostrar con distancia si está disponible
                dist_str = f", {distancia_km:.2f} km" if distancia_km > 0 else ""
                print(f"   Ruta #{self.contador_rutas}: {ruta_str} → {int(familias_quedan)} fam ({int(familias_quedan)*h}p{dist_str})")
            
            # Si hay familias que continúan, seguir explorando
            if familias_salen > 0:
                for siguiente in grafo[nodo_actual]:
                    if siguiente not in camino:  # Evitar ciclos
                        self._explorar_rutas(id_val, h, origen, camino + [siguiente], grafo, rutas_encontradas)
        else:
            # Continuar explorando desde nodo actual
            if nodo_actual in grafo:
                for siguiente in grafo[nodo_actual]:
                    if siguiente not in camino:  # Evitar ciclos
                        self._explorar_rutas(id_val, h, origen, camino + [siguiente], grafo, rutas_encontradas)
    
    def _calcular_flujo_entrante(self, camino, grafo):
        """Calcula el flujo que entra al último nodo del camino"""
        if len(camino) < 2:
            return 0
        
        nodo_anterior = camino[-2]
        nodo_actual = camino[-1]
        
        if nodo_anterior in grafo and nodo_actual in grafo[nodo_anterior]:
            return grafo[nodo_anterior][nodo_actual]
        return 0
    
    def generar_resumen(self):
        """Genera resumen estadístico de rutas"""
        print("\n" + "="*80)
        print("📊 RESUMEN DE RUTAS")
        print("="*80)
        
        if not self.rutas:
            print("No hay rutas construidas")
            return
        
        total_familias = sum(r['familias'] for r in self.rutas.values())
        total_personas = sum(r['personas'] for r in self.rutas.values())
        
        print(f"\n🎯 Totales:")
        print(f"   Rutas únicas: {len(self.rutas)}")
        print(f"   Familias evacuadas: {total_familias}")
        print(f"   Personas evacuadas: {total_personas}")
        
        # Por destino
        por_destino = defaultdict(lambda: {'familias': 0, 'personas': 0, 'rutas': 0})
        for ruta_info in self.rutas.values():
            destino = ruta_info['destino']
            por_destino[destino]['familias'] += ruta_info['familias']
            por_destino[destino]['personas'] += ruta_info['personas']
            por_destino[destino]['rutas'] += 1
        
        print(f"\n📍 Por destino:")
        for destino in sorted(por_destino.keys()):
            datos = por_destino[destino]
            print(f"   {destino}: {datos['rutas']} rutas, {datos['familias']} fam, {datos['personas']}p")
        
        # Distancias de rutas (en km si están disponibles)
        distancias = [r.get('distancia_km', 0) for r in self.rutas.values()]
        tiene_distancias = any(d > 0 for d in distancias)
        
        print(f"\n📏 Longitud de rutas:")
        if tiene_distancias:
            # Mostrar tabla detallada de rutas con distancias
            print(f"\n   {'RUTA':<25} {'FAM':>6} {'PERS':>6} {'DIST (km)':>12}")
            print(f"   {'-'*25} {'-'*6} {'-'*6} {'-'*12}")
            for ruta in sorted(self.rutas.values(), key=lambda x: x['personas'], reverse=True):
                dist_km = ruta.get('distancia_km', 0)
                print(f"   {ruta['ruta']:<25} {ruta['familias']:>6} {ruta['personas']:>6} {dist_km:>12.2f}")
            print(f"   {'-'*25} {'-'*6} {'-'*6} {'-'*12}")
            print(f"   {'TOTAL':<25} {total_familias:>6} {total_personas:>6} {sum(distancias):>12.2f}")
            
            # Estadísticas de distancia
            print(f"\n   Distancia mínima: {min(distancias):.2f} km")
            print(f"   Distancia máxima: {max(distancias):.2f} km")
            print(f"   Distancia promedio: {sum(distancias)/len(distancias):.2f} km")
            
            # Calcular personas-km
            personas_km = sum(r['personas'] * r.get('distancia_km', 0) for r in self.rutas.values())
            print(f"   Total personas-km: {personas_km:.2f}")
        else:
            # Fallback: mostrar en arcos si no hay distancias
            longitudes = [r['longitud'] for r in self.rutas.values()]
            print(f"   Mínima: {min(longitudes)} arcos")
            print(f"   Máxima: {max(longitudes)} arcos")
            print(f"   Promedio: {sum(longitudes)/len(longitudes):.1f} arcos")
            print(f"   ⚠️  Distancias en km no disponibles (pasar distancias_arcos al constructor)")
        
        # Top 5 rutas más usadas
        print(f"\n🔝 Top 5 rutas más usadas:")
        rutas_ordenadas = sorted(self.rutas.values(), key=lambda x: x['personas'], reverse=True)
        for i, ruta in enumerate(rutas_ordenadas[:5], 1):
            dist_str = f", {ruta.get('distancia_km', 0):.2f} km" if tiene_distancias else ""
            print(f"   {i}. {ruta['ruta']}: {ruta['familias']} fam ({ruta['personas']}p{dist_str})")


# ═══════════════════════════════════════════════════════════════════════════
# CASOS DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════

def caso_georeferenciado_habana():
    """
    Caso georeferenciado COMPATIBLE con PII_Flota_Asig.py (caso 'simple')
    
    Datos IDÉNTICOS al caso simple de PII_Flota_Asig.py:
    - fi_nominal con la misma estructura
    - Red: A1, A2 -> R1 -> F1, F2
    - Capacidades ajustadas
    
    Returns:
        dict con geo, arcos, rutas, solucion, etc. para usar en PII
    """
    print("=" * 80)
    print("🌍 CASO GEOREFERENCIADO: Red de Evacuación La Habana")
    print("   (COMPATIBLE con PII_Flota_Asig.py)")
    print("=" * 80)
    
    # Crear red georeferenciada
    geo, arcos = crear_red_georeferenciada_ejemplo_habana()
    
    # Obtener listas de nodos
    nodos_salida, nodos_transito, nodos_llegada = geo.obtener_listas_nodos()
    
    # ═══════════════════════════════════════════════════════════════════════
    # DATOS IDÉNTICOS AL CASO SIMPLE DE PII_Flota_Asig.py
    # ═══════════════════════════════════════════════════════════════════════
    fi_nominal = {
        (5, 'A1'): 2,   # 2 familias × 5p en A1 = 10p
        (3, 'A1'): 2,   # 2 familias × 3p en A1 = 6p
        (4, 'A1'): 1,   # 1 familia × 4p en A1 = 4p
        (5, 'A2'): 4,   # 4 familias × 5p en A2 = 20p
        (4, 'A2'): 5,   # 5 familias × 4p en A2 = 20p
    }
    # Total: 14 familias, 60 personas
    
    # CAPACIDADES GENÉRICAS: P = suma total de personas (con margen)
    P = sum(h * q for (h, ns), q in fi_nominal.items())  # 60 personas
    P_margen = int(P * 1.25)  # 25% margen para escenario pesimista
    
    capacidades = {
        ('beta', 'R1'): P_margen,           # Capacidad tránsito = P con margen
        ('pi', 'F1'): P_margen // 2,        # Capacidad neta F1 = P/2 con margen
        ('pi', 'F2'): P_margen // 2 + P_margen % 2,  # F2 = resto
        ('gamma', 'F1'): P_margen,          # Capacidad entrada = P con margen
        ('gamma', 'F2'): P_margen,
    }
    # Nota: Capacidades genéricas garantizan factibilidad para cualquier demanda ≤ P*1.25
    
    idf = generar_idf(fi_nominal)
    total_fam = sum(idf.values())
    total_pers = sum(h*q for (_,h,_),q in idf.items())
    
    print(f"\n📋 Datos de evacuación (COMPATIBLES CON PII):")
    print(f"   IDs generados: {len(idf)}")
    print(f"   Familias totales: {total_fam}")
    print(f"   Personas totales: {total_pers}")
    print(f"\n   Detalle por origen:")
    pers_a1 = sum(h*c for (h,ns),c in fi_nominal.items() if ns=='A1')
    pers_a2 = sum(h*c for (h,ns),c in fi_nominal.items() if ns=='A2')
    print(f"      A1 (Vedado): {pers_a1}p")
    print(f"      A2 (Miramar): {pers_a2}p")
    
    # Crear y resolver modelo
    modelo = ModeloAIMMS(nodos_salida, nodos_transito, nodos_llegada, arcos, idf, capacidades)
    modelo.crear_variables()
    modelo.agregar_restricciones()
    modelo.establecer_objetivo()
    
    print(f"\n🚀 Resolviendo modelo con distancias georeferenciadas...")
    status = modelo.resolver()
    print(f"   Status: {status}")
    
    if status in ['OPTIMAL', 'FEASIBLE']:
        solucion = modelo.obtener_solucion()
        print(f"   Costo total: ${solucion['objetivo']:.2f}")
        print(f"   Tiempo: {solucion['tiempo']:.3f}s")
        
        # Mostrar flujos
        print(f"\n📦 Flujos óptimos:")
        for (id_val, h, i, j), fam in sorted(solucion['X'].items()):
            nombre_i = geo.nodos[i].nombre[:12] if i in geo.nodos else i
            nombre_j = geo.nodos[j].nombre[:12] if j in geo.nodos else j
            dist = arcos.get((i, j), 0)
            print(f"   {i}→{j}: {fam} fam ({fam*h}p) | {dist:.1f}km | {nombre_i}→{nombre_j}")
        
        # Construir rutas
        constructor = ConstructorRutas(
            solucion['X'], idf,
            nodos_salida, nodos_transito, nodos_llegada, 
            list(arcos.keys()),  # Lista de arcos (tuplas)
            distancias_arcos=arcos  # Diccionario con distancias
        )
        rutas = constructor.construir_rutas()
        constructor.generar_resumen()
        
        # Retornar datos estructurados para uso en PII_Flota_Asig
        return {
            'geo': geo,
            'arcos': arcos,
            'rutas': rutas,
            'solucion': solucion,
            'nodos_salida': nodos_salida,
            'nodos_transito': nodos_transito,
            'nodos_llegada': nodos_llegada,
            'fi_nominal': fi_nominal,
            'capacidades': capacidades,
            'idf': idf
        }
    else:
        print("❌ Modelo infeasible")
        return None


def caso_complejo():
    """
    Caso complejo con nodos R intermedios y divisiones de flujo
    (CASO ORIGINAL - sin cambios)
    """
    print("="*80)
    print("🧪 CASO COMPLEJO: Red con nodos R y divisiones de flujo")
    print("="*80)
    
    fi_nominal = {
        (5, 'A1'): 2,
        (5, 'A2'): 1,
        (3, 'A1'): 3,
        (2, 'A2'): 4,
    }
    
    nodos_salida = ['A1', 'A2']
    nodos_transito = ['R1', 'R2']
    nodos_llegada = ['F1', 'F2']
    
    arcos = {
        ('A1', 'R1'): 1.0,
        ('A2', 'R1'): 1.5,
        ('R1', 'F1'): 2.0,
        ('R1', 'R2'): 1.0,
        ('R2', 'F2'): 1.5,
        ('A1', 'F2'): 3.0,
    }
    
    capacidades = {
        ('beta', 'R1'): 50,
        ('beta', 'R2'): 50,
        ('pi', 'F1'): 20,
        ('pi', 'F2'): 20,
        ('gamma', 'F1'): 100,
        ('gamma', 'F2'): 100,
    }
    
    idf = generar_idf(fi_nominal)
    total_fam = sum(idf.values())
    total_pers = sum(h*q for (_,h,_),q in idf.items())
    print(f"\n📋 Datos: {len(idf)} IDs, {total_fam} familias, {total_pers} personas")
    
    modelo = ModeloAIMMS(nodos_salida, nodos_transito, nodos_llegada, arcos, idf, capacidades)
    modelo.crear_variables()
    print(f"   Variables: {len(modelo.X)} X, {len(modelo.Y)} Y")
    
    modelo.agregar_restricciones()
    print(f"   Restricciones agregadas")
    
    modelo.establecer_objetivo()
    
    print(f"\n🚀 Resolviendo...")
    status = modelo.resolver()
    print(f"   Status: {status}")
    
    if status in ['OPTIMAL', 'FEASIBLE']:
        solucion = modelo.obtener_solucion()
        print(f"   Costo: ${solucion['objetivo']:.2f}")
        print(f"   Tiempo: {solucion['tiempo']:.3f}s")
        
        print(f"\n📦 Flujos (X > 0):")
        for (id_val, h, i, j), fam in sorted(solucion['X'].items()):
            print(f"   X[({id_val},{h},'{i}','{j}')] = {fam} fam ({fam*h}p)")
        
        constructor = ConstructorRutas(
            solucion['X'], idf,
            nodos_salida, nodos_transito, nodos_llegada, arcos
        )
        
        rutas = constructor.construir_rutas()
        constructor.generar_resumen()
        
        return rutas
    else:
        print("❌ Modelo infeasible")
        return None


def caso_simple_validacion():
    """Caso simple de referencia (120 personas) - SIN CAMBIOS"""
    print("="*80)
    print("✅ CASO SIMPLE: Validación básica (referencia)")
    print("="*80)
    
    fi_nominal = {
        (1, 'A1'): 10,
        (2, 'A2'): 10,
        (4, 'A1'): 15,
        (5, 'A1'): 4,
        (5, 'A2'): 2,
    }
    
    nodos_salida = ['A1', 'A2']
    nodos_transito = []
    nodos_llegada = ['F1', 'F2']
    
    arcos = {
        ('A1', 'F1'): 1.0,
        ('A2', 'F2'): 1.0,
        ('F1', 'A2'): 2.0,
    }
    
    capacidades = {
        ('pi', 'F1'): 50,
        ('pi', 'F2'): 70,
        ('gamma', 'F1'): 120,
        ('gamma', 'F2'): 120,
    }
    
    idf = generar_idf(fi_nominal)
    print(f"📋 Datos: {len(idf)} IDs, {sum(idf.values())} fam, {sum(h*q for (_,h,_),q in idf.items())}p")
    
    modelo = ModeloAIMMS(nodos_salida, nodos_transito, nodos_llegada, arcos, idf, capacidades)
    modelo.crear_variables()
    modelo.agregar_restricciones()
    modelo.establecer_objetivo()
    
    status = modelo.resolver()
    
    if status in ['OPTIMAL', 'FEASIBLE']:
        solucion = modelo.obtener_solucion()
        print(f"✅ Status: {status}, Costo: ${solucion['objetivo']:.2f}")
        
        constructor = ConstructorRutas(
            solucion['X'], idf,
            nodos_salida, nodos_transito, nodos_llegada, arcos
        )
        constructor.construir_rutas()
        constructor.generar_resumen()
    else:
        print("❌ Caso simple falló")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Validar caso simple (sin cambios)
    caso_simple_validacion()
    
    print("\n\n")
    
    # 2. Probar caso complejo (sin cambios)
    caso_complejo()
    
    print("\n\n")
    
    # 3. NUEVO: Probar caso georeferenciado
    caso_georeferenciado_habana()