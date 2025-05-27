                        fig = px.bar(
                            growth_df,
                            x='Ticker',
                            y='YoY Growth (%)',
                            title=f"Year-over-Year TTM Growth: {format_metric_name(ttm_metric)}",
                            color='YoY Growth (%)',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Insufficient data for TTM growth analysis. Need at least 5 quarters of data.")
    
    with tab7:
        st.header("Valuation Analysis")
        
        if not selected_tickers:
            st.warning("Please select tickers to view valuation analysis.")
        else:
            # Valuation Multiples Summary
            st.subheader("Valuation Multiples Summary")
            multiple_summary = create_multiple_summary_table(df, selected_tickers)
            if not multiple_summary.empty:
                st.dataframe(multiple_summary, use_container_width=True)
                
                st.markdown("**Legend**: P/E TTM = Price / EPS TTM, P/S TTM = Price / Revenue TTM")
            else:
                st.warning("No valuation multiple data available.")
            
            # Current Valuation Multiples
            st.subheader("Current Valuation Multiples")
            filtered_df = df[df['Ticker'].isin(selected_tickers)]
            
            if not filtered_df.empty and 'Date' in filtered_df.columns:
                # Get most recent quarter for each ticker
                latest_data = filtered_df.loc[filtered_df.groupby('Ticker')['Date'].idxmax()]
                
                # Select multiple columns
                multiple_cols = [col for col in latest_data.columns if col in ['PE_TTM', 'PS_TTM']]
                display_cols = ['Ticker', 'Report', 'Date', 'Price', 'EPS_TTM', 'Revenue_TTM'] + multiple_cols
                
                current_multiples = latest_data[display_cols].round(2)
                st.dataframe(current_multiples, use_container_width=True)
            
            # P/E TTM Analysis
            st.subheader("P/E TTM Analysis")
            if 'PE_TTM' in metrics_dict['multiple_metrics']:
                # P/E TTM comparison chart
                pe_data = filtered_df.groupby('Ticker')['PE_TTM'].mean().sort_values(ascending=True)
                pe_data = pe_data.dropna()
                
                if not pe_data.empty:
                    pe_chart_data = pd.DataFrame({
                        'Ticker': pe_data.index,
                        'Average_PE_TTM': pe_data.values
                    })
                    
                    fig = px.bar(
                        pe_chart_data,
                        x='Ticker',
                        y='Average_PE_TTM',
                        title="Average P/E Ratio (TTM) by Ticker",
                        labels={'Average_PE_TTM': 'P/E Ratio (TTM)'},
                        color='Average_PE_TTM',
                        color_continuous_scale='RdYlBu_r'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # P/E TTM insights
                    st.markdown("**P/E TTM Insights:**")
                    lowest_pe = pe_data.iloc[0]
                    highest_pe = pe_data.iloc[-1]
                    median_pe = pe_data.median()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lowest P/E TTM", f"{pe_data.index[0]}: {lowest_pe:.1f}x")
                    with col2:
                        st.metric("Median P/E TTM", f"{median_pe:.1f}x")
                    with col3:
                        st.metric("Highest P/E TTM", f"{pe_data.index[-1]}: {highest_pe:.1f}x")
            
            # P/S TTM Analysis
            st.subheader("P/S TTM Analysis")
            if 'PS_TTM' in metrics_dict['multiple_metrics']:
                # P/S TTM comparison chart
                ps_data = filtered_df.groupby('Ticker')['PS_TTM'].mean().sort_values(ascending=True)
                ps_data = ps_data.dropna()
                
                if not ps_data.empty:
                    ps_chart_data = pd.DataFrame({
                        'Ticker': ps_data.index,
                        'Average_PS_TTM': ps_data.values
                    })
                    
                    fig = px.bar(
                        ps_chart_data,
                        x='Ticker',
                        y='Average_PS_TTM',
                        title="Average P/S Ratio (TTM) by Ticker",
                        labels={'Average_PS_TTM': 'P/S Ratio (TTM)'},
                        color='Average_PS_TTM',
                        color_continuous_scale='RdYlBu_r'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # P/S TTM insights
                    st.markdown("**P/S TTM Insights:**")
                    lowest_ps = ps_data.iloc[0]
                    highest_ps = ps_data.iloc[-1]
                    median_ps = ps_data.median()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lowest P/S TTM", f"{ps_data.index[0]}: {lowest_ps:.1f}x")
                    with col2:
                        st.metric("Median P/S TTM", f"{median_ps:.1f}x")
                    with col3:
                        st.metric("Highest P/S TTM", f"{ps_data.index[-1]}: {highest_ps:.1f}x")
            
            # Valuation vs Growth Analysis
            st.subheader("Valuation vs Growth Analysis")
            if 'PE_TTM' in filtered_df.columns and 'EPS_TTM' in filtered_df.columns:
                # Create scatter plot of P/E TTM vs EPS TTM Growth
                growth_valuation_data = []
                
                for ticker in selected_tickers:
                    ticker_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values('Date')
                    if len(ticker_data) >= 5:  # Need at least 5 quarters
                        current_pe = ticker_data['PE_TTM'].iloc[-1]
                        current_eps_ttm = ticker_data['EPS_TTM'].iloc[-1]
                        year_ago_eps_ttm = ticker_data['EPS_TTM'].iloc[-5]
                        
                        if pd.notna(current_pe) and pd.notna(current_eps_ttm) and pd.notna(year_ago_eps_ttm) and year_ago_eps_ttm != 0:
                            eps_growth = ((current_eps_ttm - year_ago_eps_ttm) / year_ago_eps_ttm) * 100
                            growth_valuation_data.append({
                                'Ticker': ticker,
                                'PE_TTM': current_pe,
                                'EPS_Growth_YoY': eps_growth
                            })
                
                if growth_valuation_data:
                    growth_val_df = pd.DataFrame(growth_valuation_data)
                    
                    fig = px.scatter(
                        growth_val_df,
                        x='EPS_Growth_YoY',
                        y='PE_TTM',
                        text='Ticker',
                        title="P/E TTM vs EPS Growth (YoY)",
                        labels={
                            'EPS_Growth_YoY': 'EPS Growth YoY (%)',
                            'PE_TTM': 'P/E Ratio (TTM)'
                        }
                    )
                    fig.update_traces(textposition="top center")
                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Interpretation:**
                    - **Lower Left Quadrant**: Low growth, low valuation (Value stocks)
                    - **Lower Right Quadrant**: High growth, low valuation (Growth at reasonable price)
                    - **Upper Left Quadrant**: Low growth, high valuation (Potentially overvalued)
                    - **Upper Right Quadrant**: High growth, high valuation (Growth stocks)
                    """)
                else:
                    st.warning("Insufficient data for valuation vs growth analysis.")

if __name__ == "__main__":
    main()
