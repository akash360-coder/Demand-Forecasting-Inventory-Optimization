from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


class Product(Base):
    __tablename__ = "products"

    product_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(100), nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sales = relationship("Sales", back_populates="product")
    inventory = relationship("Inventory", back_populates="product")
    forecasts = relationship("Forecast", back_populates="product")
    recommendation_records = relationship("InventoryRecommendation", back_populates="product")


class Store(Base):
    __tablename__ = "stores"

    store_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    region = Column(String(100), nullable=False)
    city = Column(String(100), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sales = relationship("Sales", back_populates="store")
    inventory = relationship("Inventory", back_populates="store")
    forecasts = relationship("Forecast", back_populates="store")
    recommendation_records = relationship("InventoryRecommendation", back_populates="store")


class Sales(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    product_id = Column(String(64), ForeignKey("products.product_id"), nullable=False, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    units_sold = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    promotion = Column(Float, default=0.0)
    holiday = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="sales")
    store = relationship("Store", back_populates="sales")


class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(64), ForeignKey("products.product_id"), nullable=False, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    inventory_on_hand = Column(Float, nullable=False, default=0.0)
    lead_time_days = Column(Float, nullable=False, default=0.0)
    safety_stock = Column(Float, default=0.0)
    reorder_point = Column(Float, default=0.0)
    target_inventory = Column(Float, default=0.0)
    abc_class = Column(String(10), default="C")
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="inventory")
    store = relationship("Store", back_populates="inventory")


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(64), ForeignKey("products.product_id"), nullable=False, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    target_date = Column(Date, nullable=False, index=True)
    model_name = Column(String(80), nullable=False)
    model_version = Column(String(80), nullable=True)
    dataset_version = Column(String(80), nullable=True)
    feature_list = Column(Text, nullable=True)
    predicted_demand = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="forecasts")
    store = relationship("Store", back_populates="forecasts")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(80), nullable=False, index=True)
    model_version = Column(String(80), nullable=True)
    dataset_version = Column(String(80), nullable=True)
    metric_name = Column(String(40), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    evaluation_period = Column(String(80), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    product_id = Column(String(64), ForeignKey("products.product_id"), nullable=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=True, index=True)


class InventoryRecommendation(Base):
    __tablename__ = "inventory_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(64), ForeignKey("products.product_id"), nullable=False, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    recommendation_date = Column(Date, nullable=False, index=True)
    demand_during_lead_time = Column(Float, default=0.0)
    demand_variability = Column(Float, default=0.0)
    safety_stock = Column(Float, default=0.0)
    reorder_point = Column(Float, default=0.0)
    target_inventory = Column(Float, default=0.0)
    recommended_order_quantity = Column(Float, default=0.0)
    stockout_risk = Column(Float, default=0.0)
    excess_inventory_risk = Column(Float, default=0.0)
    stockout_label = Column(String(20), default="LOW")
    excess_label = Column(String(20), default="LOW")
    abc_class = Column(String(10), default="C")
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="recommendation_records")
    store = relationship("Store", back_populates="recommendation_records")
