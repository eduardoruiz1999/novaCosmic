# Para el Sol (monitoreo central)
solar_monitor = CosmicNetworkMonitor("finney")
asyncio.create_task(solar_monitor.start_comprehensive_monitoring())

# Para planetas individuales (consultas específicas)
earth_explorer = CosmicSubnetExplorer("finney")
subnets = await earth_explorer.discover_all_subnets_async()
